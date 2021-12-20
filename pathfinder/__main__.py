import os
import json
import yaml

import click
import numpy as np

from .openlane import metrics
from . import model


@click.group()
def cli():
    pass


@click.command("pretrain")
def pretrain():
    model.pretrain()


cli.add_command(pretrain)


@click.command("train")
@click.argument("presynth_cache_file")
def train(presynth_cache_file):
    model.train(presynth_cache_file)


cli.add_command(train)


@click.command("presynthesize")
@click.argument("presynth_cache_file")
def presynthesize(presynth_cache_file):
    model.presynthesize_designs(presynth_cache_file)


cli.add_command(presynthesize)


@click.command("predict")
@click.option("--normalize/--no-normalize", default=True)
@click.option("--output-type", type=click.Choice(["tcl", "yaml"]), default="yaml")
@click.argument("inputs_file")
def predict(normalize, inputs_file, output_type):
    data_str = open(inputs_file).read()

    data = {}

    if inputs_file.endswith(".json"):
        data = json.loads(data_str)
    else:
        data = yaml.safe_load(data_str)

    if normalize:
        data = metrics.normalize_input_metrics(data)

    model_to_load = model.train_best_model_path
    if not os.path.exists(model_to_load):
        model_to_load = model.pretrain_best_model_path

    loaded_model = model.PathfinderModel.by_loading_weights(model_to_load)

    input_array = np.array(list(data.values()))
    data_for_prediction = np.expand_dims(input_array, axis=0)

    prediction = loaded_model.predict(data_for_prediction)
    prediction_dict = metrics.output_array_to_dict(prediction[0].tolist())
    prediction_denormalized = metrics.denormalize_output_variables(prediction_dict)

    if output_type == "tcl":
        for key, value in prediction_denormalized.items():
            print(f"set ::env({key}) {{{value}}}")
    else:
        print(yaml.safe_dump(prediction_denormalized))


cli.add_command(predict)

if __name__ == "__main__":
    cli()
