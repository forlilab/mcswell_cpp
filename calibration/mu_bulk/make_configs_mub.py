#!/usr/bin/env python
"""Configs for the mu_bulk calibration titrations (see make_water_boxes.py and README.md).

For each water model, snapshot and cube size (boxes.json) and starting state (empty, full)
one run: the central cube is titrated from mu = -10 to -2 kcal/mol in 0.25 steps, with the shell
waters fixed (receptor.fixed_waters) and no protein. Grid spacing is the production one
(0.375 A); steps are twice the production ones (200M + 40M equilibration), because at 1/100
of the production steps a 20 A cube was still filling during production (see README.md). distance_cutoff only limits where
insertion points go (within that distance of a fixed atom), so it is 30 A here to cover the
whole cube. 2 models x 3 snapshots x 3 cube sizes x 2 starts = 36 runs, one per line of tasks.txt.

    python make_configs_mub.py                          # runs and shells next to this script
    python make_configs_mub.py --root /path/as/seen/by/the/jobs/calibration/mu_bulk
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

MU_RANGE = {"start": -10.0, "stop": -2.0, "step": 0.25}
FILLS = ("empty", "full")


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML: {v!r}")


def dump_toml(config):
    """Minimal writer for these configs: top-level scalars and one level of tables."""
    lines = [f"{k} = {toml_value(v)}" for k, v in config.items() if not isinstance(v, dict)]
    for table, values in config.items():
        if isinstance(values, dict):
            lines += ["", f"[{table}]"] + [f"{k} = {toml_value(v)}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=HERE, help="This folder as seen by the jobs (holds shells/, gets runs/).")
    args = ap.parse_args()
    boxes = json.load(open(os.path.join(HERE, "boxes.json")))
    os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
    tasks = []
    for model, snaps in boxes.items():
        for s in snaps:
            for f, fill in enumerate(FILLS):
                name = f"{model}_box{s['snapshot']}_r{s['region_A']:g}_{fill}"
                cx, cy, cz = s["center_A"]
                cfg = {
                    "title": "MCSwell mu_bulk calibration: titration of a water cube in a fixed water shell",
                    "io": {"save_path": f"{args.root}/runs/{name}"},
                    "receptor": {"fixed_waters": f"{args.root}/{s['shell']}"},
                    "simulation_parameters": {
                        "n_frames": 1, "n_snapshots": 1000,
                        "n_equilibration_steps": 40000000, "n_gcmc_steps": 200000000,
                        "distance_cutoff": 30.0, "initial_fill": fill,
                        "seed": 7300001 + 1000 * int(s["region_A"]) + 100 * s["snapshot"] + 10 * f
                                + (1 if model == "tip4p" else 0),
                    },
                    "mu_range": dict(MU_RANGE),
                    "simulation_box": {"spacing": 0.375, "center_x": cx, "center_y": cy, "center_z": cz,
                                       "x_size": s["region_A"], "y_size": s["region_A"],
                                       "z_size": s["region_A"]},
                }
                with open(os.path.join(HERE, "configs", f"{name}.toml"), "w") as fo:
                    fo.write(dump_toml(cfg))
                tasks.append(f"{model} {name}")
    with open(os.path.join(HERE, "tasks.txt"), "w") as fo:
        fo.write("\n".join(tasks) + "\n")
    print(f"{len(tasks)} runs in tasks.txt")


if __name__ == "__main__":
    main()
