# Needs unified tarball from OpenLane. Not a thing anymore
import yaml
import subprocess
from pathfinder.openlane import *

def run(cmd: str):
    print(f"$ {cmd}")
    op = subprocess.check_output(cmd, shell=True).decode("utf8")
    print(op, end="")
    return op

tarballs_raw = run(f"find ~/Downloads/rt/ | grep tar.gz")
tarballs = tarballs_raw.rstrip().split("\n")

final_array = []

for tarball in tarballs:
    folder = tarball[:-7]
    run(f"rm -rf {folder} && mkdir -p {folder}")
    run(f"tar -xf {tarball} -C {folder} --strip-components=6")
    all = metrics.get_rundir_metrics(folder)
    if "failed" in all["flow_status"]:
        print(f"Skipping {tarball}...")
        continue
    envs = metrics.get_rundir_envs(folder)

    inputs = metrics.isolate_input_metrics(all)
    outputs = metrics.isolate_output_variables(envs)

    inputs_n = metrics.normalize_input_metrics(inputs)
    outputs_n = metrics.normalize_output_variables(outputs)

    final_array.append({"inputs": inputs_n, "outputs": outputs_n})

with open("./data/pretrain.yml", "w") as f:
    f.write(yaml.dump(final_array, sort_keys=False))
