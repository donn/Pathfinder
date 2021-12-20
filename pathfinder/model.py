import os
import sys
import yaml
import pathlib
from typing import List, Tuple, Dict, Union

import numpy as np
import tensorflow as tf

from . import openlane

keras = tf.keras

model_dir = f"{os.path.dirname(os.path.dirname(__file__))}/models"
pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
pretrain_best_model_path = f"{model_dir}/pretrain_best.h5"
train_best_model_path = f"{model_dir}/train_best.h5"

design_dir = f"{os.path.dirname(os.path.dirname(__file__))}/designs"


class PathfinderModel(keras.Model):
    def __init__(self):
        L = keras.layers
        inputs = L.Input(name="input", shape=(14,))
        layer1 = L.Dense(name="hl1", units=16, activation="sigmoid")(inputs)
        layer2 = L.Dense(name="hl2", units=16, activation="sigmoid")(layer1)
        outputs = L.Dense(name="output", units=8, activation="sigmoid")(layer2)
        super().__init__(inputs=inputs, outputs=outputs)

    def compiled(
        self,
        optimizer=keras.optimizers.SGD(learning_rate=0.02),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
        **kwargs,
    ):
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        return self

    @staticmethod
    def by_loading_weights(path: str) -> "PathfinderModel":
        model = PathfinderModel().compiled()
        model.load_weights(path)
        return model


class CriticModel(keras.Model):
    """
    Takes action from reward and predicts the punishment.
    """

    def __init__(self):
        L = keras.layers
        inputs = L.Input(name="input", shape=(8,))
        layer1 = L.Dense(name="hl1", units=16, activation="ReLU")(inputs)
        layer2 = L.Dense(name="hl2", units=16, activation="ReLU")(layer1)
        output = L.Dense(name="q", units=1, activation="ReLU")(layer2)
        super().__init__(inputs=inputs, outputs=output)


class CallbackAccuracyGTE(keras.callbacks.Callback):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get("accuracy")
        if current is None:
            return

        if current >= self.value:
            self.model.stop_training = True


def pretrain(min_accuracy=0.7):
    accuracy = 0.0
    while accuracy < min_accuracy:
        model = PathfinderModel().compiled()
        data = yaml.safe_load(open("./data/pretrain.yml").read())

        train_data = list(map(lambda x: list(x["inputs"].values()), data))
        train_labels = list(map(lambda x: list(x["outputs"].values()), data))

        checkpoint = keras.callbacks.ModelCheckpoint(
            pretrain_best_model_path,
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="accuracy",
            min_delta=0.1,
            patience=50,
            mode="max",
            restore_best_weights=True,
        )
        stop_at_min = CallbackAccuracyGTE(value=min_accuracy)

        model.fit(
            train_data,
            train_labels,
            epochs=500,
            callbacks=[checkpoint, early_stop, stop_at_min],
        )

        model = PathfinderModel.by_loading_weights(pretrain_best_model_path)

        test_loss, test_acc = model.evaluate(train_data, train_labels, verbose=2)

        print(f"Loss: {test_loss}, Accuracy: {test_acc}", file=sys.stderr)

        accuracy = test_acc


design_dict_type = Dict[str, Dict[str, Union[str, np.array]]]


def presynthesize_designs(presynth_cache_file):
    olm = openlane.metrics

    design_dict: design_dict_type = {}
    print("Synthesizing all designs...")
    for design_name in os.listdir(design_dir):
        print(f"\tSynthesizing design {design_name}...")
        design = os.path.join(design_dir, design_name)
        run_tag, inputs = olm.run_and_get_input_metrics(design)
        normalized_inputs = olm.normalize_input_metrics(inputs)
        inputs_as_array = np.array(list(normalized_inputs.values()))
        input_tensors = np.expand_dims(inputs_as_array, axis=0)
        design_dict[design_name] = {
            "run_tag": run_tag,
            "input_tensors": input_tensors.tolist(),
        }
        with open(presynth_cache_file, "w") as f:
            f.write(yaml.safe_dump(design_dict, sort_keys=False))


def train(presynth_cache_file):
    # Adapted from https://keras.io/examples/rl/actor_critic_cartpole/
    olm = openlane.metrics

    f32_epsilon = np.finfo(np.float32).eps.item()

    model = PathfinderModel.by_loading_weights(pretrain_best_model_path)
    model_optimizer = keras.optimizers.Adam(learning_rate=0.1)

    critic = CriticModel()
    critic_optimizer = keras.optimizers.Adam(learning_rate=0.1)
    critic_loss = keras.losses.Huber()

    design_dict: design_dict_type = yaml.safe_load(open(presynth_cache_file).read())

    for design, values in design_dict.items():
        values["input_tensors"] = np.array(values["input_tensors"])

    design_list = os.listdir(design_dir)

    punishment_running_average: float = -600 * len(design_list)
    ra_window = 5
    ra_ratio = 1 / ra_window

    episode_punishments_abs = []

    for episode in range(0, 5):  # Up to N episodes
        episode_punishment: float = 0

        critic_value_history = []
        punishments_history = []

        print(f"Starting episode {episode}...")
        with tf.GradientTape(persistent=True) as tape:
            for design_name in os.listdir(design_dir):
                print(f"\tRunning design {design_name}...")

                run_tag, input_tensors = design_dict[design_name].values()

                design = os.path.join(design_dir, design_name)

                out_tensor = model(input_tensors)
                critic_value = critic(out_tensor)

                proposed_variables = out_tensor[0]
                proposed_variables_dict = olm.output_array_to_dict(proposed_variables)
                proposed_variables_denormalized = olm.denormalize_output_variables(
                    proposed_variables_dict
                )

                tns = olm.run_and_quantify_closure(
                    design,
                    run_tag,
                    proposed_variables_denormalized,
                    f"e{episode}d{design_name}",
                )

                punishment = tns * 100 - np.sum(proposed_variables * 100)
                print(f"\tDone {design_name}: tns {tns}, punishment {punishment}")

                punishments_history.append(punishment)
                critic_value_history.append(critic_value[0, 0])

                episode_punishment += punishment

            episode_punishments_abs.append(str(abs(episode_punishment.item())))
            with open("pra.csv", "w") as f:
                f.write(",".join(episode_punishments_abs))

            punishment_running_average = (
                episode_punishment * ra_ratio
                + punishment_running_average * (1 - ra_ratio)
            )
            print(
                f"Episode {episode} punishment: {episode_punishment}, Punishment Running Avg.: {punishment_running_average}"
            )

            returns = []
            discounted_sum = 0
            for r in punishments_history[::-1]:
                discounted_sum = r + 0.99 * discounted_sum
                returns.insert(0, discounted_sum)

            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + f32_epsilon)
            returns = returns.tolist()

            actor_losses = []
            critic_losses = []
            for critic_value, return_value in zip(critic_value_history, returns):
                diff = return_value - critic_value
                actor_losses.append(-diff)
                critic_losses.append(
                    critic_loss(
                        tf.expand_dims(critic_value, 0), tf.expand_dims(return_value, 0)
                    )
                )

            loss_total = sum(actor_losses) + sum(critic_losses)

        model_grads = tape.gradient(loss_total, model.trainable_variables)
        model_optimizer.apply_gradients(zip(model_grads, model.trainable_variables))

        critic_grads = tape.gradient(loss_total, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

        del tape
