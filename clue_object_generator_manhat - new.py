#!/usr/bin/env python3
"""
Option 1 generator (with metadata header):
- "objects" subcommand: create a master list of object locations (the seeds)
- "clues" subcommand:  re-generate clues per condition while keeping objects fixed
  Each clues CSV includes a descriptive header block.
"""

import argparse
import csv
import json
import random
from hashlib import sha256
from typing import List, Tuple, Dict

GRID_SIZE = 30
NUM_TRIALS = 50
DEFAULT_DISTRIBUTION_TYPE = 3  # 1, 2, or 3
# Distribution modes:
#1 for tightest, 2 for medium, 3 for widest
WEIGHTING_EXPONENTS = {1: 2, 2: 1, 3: .7}
clues_per_object = 2

# Hardcoded configuration - edit these to change behavior
COMMAND = "clues"  # "objects" or "clues"
OBJECTS_FILE = "objects_g30_n50.csv"  # only used when COMMAND = "clues"
SEED = None  # None or an integer
OUTPUT_FILE = None  # None to use default naming


def _stable_seed(*parts: str) -> int:
    s = ("|".join(str(p) for p in parts)).encode("utf-8")
    return int.from_bytes(sha256(s).digest()[:8], "big", signed=False)


def weighted_clue_locations(obj, cells, clues_per_object, mode, rng):
    exp = WEIGHTING_EXPONENTS[mode]
    def w(r): return 1 / ((1 + r) ** exp)
    available = [c for c in cells if c != obj]
    weights = [w(abs(c[0]-obj[0]) + abs(c[1]-obj[1])) for c in available]
    clues = []
    while len(clues) < clues_per_object and available:
        pick = rng.choices(available, weights=weights, k=1)[0]
        idx = available.index(pick)
        clues.append(available.pop(idx))
        weights.pop(idx)
    return clues


def generate_objects(grid_size, num_trials, base_seed):
    rng = random.Random(base_seed)
    return [(rng.randrange(grid_size), rng.randrange(grid_size)) for _ in range(num_trials)]


def generate_clues_for_objects(objects, grid_size, clues_per_object, distribution, base_seed):
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    trials = []
    cond_tag = f"d{distribution}_c{clues_per_object}_g{grid_size}"
    for ep, obj in enumerate(objects):
        ep_seed = _stable_seed(base_seed or "none", cond_tag, f"ep{ep}", f"obj{obj}")
        rng = random.Random(ep_seed)
        clues = weighted_clue_locations(obj, cells, clues_per_object, distribution, rng)
        trials.append({"object": obj, "clues": clues})
    return trials


def write_objects_csv(objects, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "object_x", "object_y"])
        for i, (ox, oy) in enumerate(objects):
            w.writerow([i, ox, oy])


def read_objects_csv(filename):
    objs = []
    with open(filename, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            objs.append((int(row["object_x"]), int(row["object_y"])))
    return objs


def write_trials_csv(data, filename, clues_per_object, distribution, grid_size, base_seed):
    with open(filename, "w", newline="") as f:
        # metadata header
        f.write(f"# condition: distribution={distribution}, clues_per_object={clues_per_object}, grid_size={grid_size}\n")
        f.write(f"# base_seed: {base_seed}\n")

        header = ["episode", "object_x", "object_y"]
        for i in range(clues_per_object):
            header += [f"clue{i+1}_x", f"clue{i+1}_y"]
        w = csv.writer(f)
        w.writerow(header)

        for ep, trial in enumerate(data):
            row = [ep, trial["object"][0], trial["object"][1]]
            for clue in trial["clues"]:
                row += [clue[0], clue[1]]
            # pad if needed
            while len(trial["clues"]) < clues_per_object:
                row += ["", ""]
                trial["clues"].append(("", ""))
            w.writerow(row)


def main():
    # Use hardcoded variables instead of command-line arguments
    if COMMAND == "objects":
        objs = generate_objects(GRID_SIZE, NUM_TRIALS, SEED)
        out = OUTPUT_FILE if OUTPUT_FILE else f"objects_g{GRID_SIZE}_n{NUM_TRIALS}.csv"
        write_objects_csv(objs, out)
        print(f"[objects] wrote {len(objs)} objects to {out}")
        return

    if COMMAND == "clues":
        objs = read_objects_csv(OBJECTS_FILE)
        trials = generate_clues_for_objects(objs, GRID_SIZE, clues_per_object, DEFAULT_DISTRIBUTION_TYPE, SEED)
        cond = f"d{DEFAULT_DISTRIBUTION_TYPE}_c{clues_per_object}_g{GRID_SIZE}"
        out = OUTPUT_FILE if OUTPUT_FILE else f"trials_{cond}.csv"
        write_trials_csv(trials, out, clues_per_object, DEFAULT_DISTRIBUTION_TYPE, GRID_SIZE, SEED)
        print(f"[clues] wrote {len(trials)} trials to {out} with header metadata")


if __name__ == "__main__":
    main()
