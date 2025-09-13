"""Generate random clue and object locations for grid-based trials."""

import argparse
import json
import math
import random
from typing import List, Tuple, Dict


def weighted_clue_locations(
    obj: Tuple[int, int],
    cells: List[Tuple[int, int]],
    clues_per_object: int,
) -> List[Tuple[int, int]]:
    """Return `clues_per_object` unique clue locations weighted by distance.

    The weight for a cell at Euclidean distance ``r`` from the object is
    ``1 / (1 + r)``, making nearby cells more likely.
    """

    weights = [1 / (1 + math.hypot(cx - obj[0], cy - obj[1])) for cx, cy in cells]
    available = cells.copy()
    weights_copy = weights.copy()
    clues: List[Tuple[int, int]] = []
    while len(clues) < clues_per_object and available:
        clue = random.choices(available, weights=weights_copy, k=1)[0]
        clues.append(clue)
        idx = available.index(clue)
        available.pop(idx)
        weights_copy.pop(idx)
    return clues


def generate_trials(
    grid_size: int = 10,
    num_trials: int = 33,
    clues_per_object: int = 2,
) -> List[Dict[str, List[Tuple[int, int]]]]:
    """Generate trial data for object and clue locations."""

    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    trials = []
    for _ in range(num_trials):
        obj = (random.randrange(grid_size), random.randrange(grid_size))
        clues = weighted_clue_locations(obj, cells, clues_per_object)
        trials.append({"object": obj, "clues": clues})
    return trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clue and object generator")
    parser.add_argument("--grid-size", type=int, default=10, help="grid dimension")
    parser.add_argument("--trials", type=int, default=33, help="number of trials")
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data = generate_trials(args.grid_size, args.trials)
    print(json.dumps(data))
