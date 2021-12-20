import os
import re
import sys
import pathlib
import subprocess

pdk_root = os.getenv("PDK_ROOT")
pdk = "sky130A"
scl = "sky130A_fd_sc_hd"

log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)

pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)


def rp(path):
    return os.path.realpath(path)


def openlane(*args_tuple, tag="ol_run"):
    args = list(args_tuple)

    stdout_f = open(os.path.join(log_dir, f"{tag}.stdout"), "w")
    stderr_f = open(os.path.join(log_dir, f"{tag}.stderr"), "w")

    env = os.environ.copy()
    env["ROUTING_CORES"] = 3

    cmd = ["/home/donn/efabless/openlane/flow.tcl"] + args
    status = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f)

    stdout_f.close()
    stderr_f.close()

    if status.returncode != 0:
        raise Exception(f"{args} failed with exit code {status.returncode}")


def read_env(config_path: str) -> dict:
    rx = r"\s*set\s*::env\((.+?)\)\s*(.+)"
    env = {}
    string_data = ""
    try:
        string_data = open(config_path).read()
    except FileNotFoundError:
        print(f"❌ File {config_path} not found.", file=sys.stderr)
        exit(os.EX_NOINPUT)

    # Process \ at ends of lines, remove semicolons
    entries = string_data.split("\n")
    i = 0
    while i < len(entries):
        if not entries[i].endswith("\\"):
            if entries[i].endswith(";"):
                entries[i] = entries[i][:-1]
            i += 1
            continue
        entries[i] = entries[i][:-1] + entries[i + 1]
        del entries[i + 1]

    for entry in entries:
        match = re.match(rx, entry)
        if match is None:
            continue
        name = match[1]
        value = match[2]
        # remove double quotes/{}
        value = value.strip('"')
        value = value.strip("{}")
        env[name] = value

    return env


def get_run_dir(design: str, run_tag: str) -> str:
    return f"{design}/runs/{run_tag}"


def override_env_str(override_env: dict) -> str:
    return ",".join([f"{k}={v}" for k, v in override_env.items()])
