import re
import os
import csv
import math
import glob
import shutil
import datetime
from typing import Dict, Any, Tuple, List

import numpy as np

from .run import openlane, get_run_dir, override_env_str


def process_report_csv(csv_in: str) -> Dict[str, str]:
    metric_dict = {}
    with open(csv_in) as f:
        csv_data = list(csv.reader(f, delimiter=",", quotechar="'"))
        column_count = len(csv_data[0])
        for column in range(0, column_count):
            key = csv_data[0][column]
            value = csv_data[1][column]
            metric_dict[key] = value
    return metric_dict


def get_rundir_metrics(dir: str) -> Dict[str, Any]:
    metrics = process_report_csv(f"{dir}/reports/final_summary_report.csv")
    yosys_log = open(f"{dir}/logs/synthesis/1-synthesis.log").read()

    cells = int(metrics["cells_pre_abc"])

    done = set()

    muxes = 0
    nots = 0
    ands = 0
    nands = 0
    ors = 0
    nors = 0
    xors = 0
    xnors = 0

    rx = re.compile(r"\$_(\w+)_\s+(\d+)")
    for match in rx.finditer(yosys_log):
        k, v = match[1], match[2]
        if k in done:
            break
        done.add(k)
        v = int(v)
        if k == "ANDNOT":
            ands += v
        elif k == "AND":
            ands += v
        elif k == "NAND":
            nands += v
        elif k == "ORNOT":
            ors += v
        elif k == "OR":
            ors += v
        elif k == "NOR":
            nors += v
        elif k == "MUX":
            muxes += v
        elif k == "XOR":
            xors += v
        elif k == "XNOR":
            xnors += v
        elif k == "NOT":
            nots += v

    dffs = cells - (muxes + nots + ands + nands + ors + nors + xors + xnors)

    metrics["DFF"] = dffs
    metrics["MUX"] = muxes
    metrics["NOT"] = nots
    metrics["AND"] = ands
    metrics["NAND"] = nands
    metrics["OR"] = ors
    metrics["NOR"] = nors
    metrics["XOR"] = xors
    metrics["XNOR"] = xnors

    return metrics


def get_rundir_envs(dir: str) -> Dict[str, str]:
    tcl = open(f"{dir}/config.tcl").read()
    rx = re.compile(r"set\s+::env\((\w+)\)\s+\{(.*)\}")
    env = {}
    for line in tcl.split("\n"):
        match = rx.search(line)
        if match is None:
            # print(f"Skipped {line}")
            continue
        key, value = match[1], match[2]
        env[key] = value
    return env


def get_run_metrics(design: str, run_tag: str):
    return get_rundir_metrics(get_run_dir(design, run_tag))


def isolate_input_metrics(all_metrics: dict) -> Dict[str, float]:
    final_dict = {}
    for final_name, initial_name in [
        ("area_mm2", "DIEAREA_mm^2"),
        ("cells", "cells_pre_abc"),
        ("inputs", "inputs"),
        ("outputs", "outputs"),
        ("levels", "level"),
        ("nets", "wire_bits"),
        ("dffs", "DFF"),
        ("nots", "NOT"),
        ("ands", "AND"),
        ("nands", "NAND"),
        ("ors", "OR"),
        ("nors", "NOR"),
        ("xors", "XOR"),
        ("xnors", "XNOR"),
        ("muxes", "MUX"),
    ]:
        final_dict[final_name] = float(all_metrics[initial_name])
    return final_dict


def normalize_input_metrics(metrics: dict) -> Dict[str, float]:
    area = math.sqrt(metrics["area_mm2"] / 10)
    cells = metrics["cells"] / 500000
    ios = (metrics["inputs"] + metrics["outputs"]) / 100000
    levels = metrics["levels"] / 256
    nets = metrics["nets"] / 500000

    gate_counts = np.array(
        [
            metrics["dffs"],
            metrics["nots"],
            metrics["ands"],
            metrics["nands"],
            metrics["ors"],
            metrics["nors"],
            metrics["xors"],
            metrics["xnors"],
            metrics["muxes"],
        ]
    )

    gate_total = sum(gate_counts)

    dffs, nots, ands, nands, ors, nors, xors, xnors, muxes = gate_counts / gate_total

    return {
        "area": area,
        "cells": cells,
        "ios": ios,
        "levels": levels,
        "nets": nets,
        "dffs": float(dffs),
        "nots": float(nots),
        "ands": float(ands),
        "nands": float(nands),
        "ors": float(ors),
        "nors": float(nors),
        "xors": float(xors),
        "xnors": float(xnors),
        "muxes": float(muxes),
    }


def isolate_output_variables(variables: Dict[str, str]) -> Dict[str, float]:
    final = {}
    for key in [
        "PL_RESIZER_SETUP_SLACK_MARGIN",
        "GLB_RESIZER_SETUP_SLACK_MARGIN",
        "PL_RESIZER_HOLD_SLACK_MARGIN",
        "GLB_RESIZER_HOLD_SLACK_MARGIN",
        "PL_RESIZER_SETUP_MAX_BUFFER_PERCENT",
        "GLB_RESIZER_SETUP_MAX_BUFFER_PERCENT",
        "PL_RESIZER_HOLD_MAX_BUFFER_PERCENT",
        "GLB_RESIZER_HOLD_MAX_BUFFER_PERCENT",
    ]:
        final[key] = float(variables[key])
    return final


def normalize_output_variables(variables: Dict[str, float]) -> Dict[str, float]:
    final = variables.copy()
    for key in [
        "PL_RESIZER_SETUP_MAX_BUFFER_PERCENT",
        "GLB_RESIZER_SETUP_MAX_BUFFER_PERCENT",
        "PL_RESIZER_HOLD_MAX_BUFFER_PERCENT",
        "GLB_RESIZER_HOLD_MAX_BUFFER_PERCENT",
    ]:
        final[key] = variables[key] / 100
    return final


def output_array_to_dict(variables: List[float]) -> Dict[str, float]:
    final = {}
    for key, value in zip(
        [
            "PL_RESIZER_SETUP_SLACK_MARGIN",
            "GLB_RESIZER_SETUP_SLACK_MARGIN",
            "PL_RESIZER_HOLD_SLACK_MARGIN",
            "GLB_RESIZER_HOLD_SLACK_MARGIN",
            "PL_RESIZER_SETUP_MAX_BUFFER_PERCENT",
            "GLB_RESIZER_SETUP_MAX_BUFFER_PERCENT",
            "PL_RESIZER_HOLD_MAX_BUFFER_PERCENT",
            "GLB_RESIZER_HOLD_MAX_BUFFER_PERCENT",
        ],
        variables,
    ):
        final[key] = value
    return final


def denormalize_output_variables(variables: Dict[str, float]) -> Dict[str, float]:
    final = variables.copy()
    for key in [
        "PL_RESIZER_SETUP_MAX_BUFFER_PERCENT",
        "GLB_RESIZER_SETUP_MAX_BUFFER_PERCENT",
        "PL_RESIZER_HOLD_MAX_BUFFER_PERCENT",
        "GLB_RESIZER_HOLD_MAX_BUFFER_PERCENT",
    ]:
        final[key] = variables[key] * 100
    return final


def run_and_get_input_metrics(design: str) -> Tuple[str, Dict[str, float]]:
    run_tag = f"{datetime.datetime.now().isoformat()}"

    override_env = {"QUIT_ON_TIMING_VIOLATIONS": 0}

    design_name = os.path.basename(design)

    openlane(
        "-tag",
        run_tag,
        "-design",
        design,
        "-to",
        "floorplan",
        "-override_env",
        override_env_str(override_env),
        tag=f"{design_name}_input_metrics",
    )

    metric_dict = get_run_metrics(design, run_tag)
    final_dict = isolate_input_metrics(metric_dict)
    return run_tag, final_dict


def run_and_quantify_closure(
    design: str, run_tag: str, inputs: dict, tag: str = "0"
) -> float:
    """
    Note that perhaps a little counterintuitively, the inputs to this function
    are the *output* from the neural network.
    """
    override_env = {"QUIT_ON_TIMING_VIOLATIONS": 0, **inputs}

    shutil.copytree(get_run_dir(design, run_tag), get_run_dir(design, f"{run_tag}.bk"))

    exception: Exception = None

    try:
        openlane(
            "-tag",
            run_tag,
            "-design",
            design,
            "-from",
            "placement",
            "-to",
            "placement",
            "-override_env",
            override_env_str(override_env),
            tag=tag,
        )

        metric_glob = glob.glob(
            f"{get_run_dir(design, run_tag)}/logs/placement/*-resizer.log"
        )
        print(metric_glob)
        metric_file = metric_glob[0]
        metric_file_str = open(metric_file).read()
        metric_rx = re.compile(r"tns\s+([\d.]+)")
        metric_match = metric_rx.search(metric_file_str)
        spef_tns = float(metric_match[1])
    except Exception as e:
        exception = e
    finally:
        final_dir = get_run_dir(design, f"{run_tag}.exploration{tag}")
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.move(get_run_dir(design, run_tag), final_dir)

        shutil.move(get_run_dir(design, f"{run_tag}.bk"), get_run_dir(design, run_tag))

    if exception is not None:
        raise exception

    return spef_tns
