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
    mode: str = "linear",
) -> List[Tuple[int, int]]:
    """Return ``clues_per_object`` unique clue locations weighted by distance.

    ``mode`` selects the weighting scheme based on the Euclidean distance ``r``
    from the object:

    * ``"linear"``  – ``1 / (1 + r)``
    * ``"square"`` – ``1 / (1 + r**2)``
    """

    # Candidate cells exclude the object's location so it never appears as a
    # clue. We keep a parallel list of weights for sampling.
    available = [cell for cell in cells if cell != obj]

    def weight(r: float) -> float:
        if mode == "square":
            return 1 / (1 + r ** 2)
        return 1 / (1 + r)

    weights = [weight(math.hypot(cx - obj[0], cy - obj[1])) for cx, cy in available]

    clues: List[Tuple[int, int]] = []
    while len(clues) < clues_per_object and available:
        clue = random.choices(available, weights=weights, k=1)[0]
        clues.append(clue)
        idx = available.index(clue)
        available.pop(idx)
        weights.pop(idx)
    return clues


def generate_trials(
    grid_size: int = 10,
    num_trials: int = 34,
    clues_per_object: int = 2,
) -> List[Dict[str, List[Tuple[int, int]]]]:
    """Generate trial data for object and clue locations.

    The first 17 trials weight clues by ``1 / (1 + r)`` and the remaining 17 by
    ``1 / (1 + r**2)``.
    """

    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    trials = []
    for i in range(num_trials):
        obj = (random.randrange(grid_size), random.randrange(grid_size))
        mode = "linear" if i < 17 else "square"
        clues = weighted_clue_locations(obj, cells, clues_per_object, mode)
        trials.append({"object": obj, "clues": clues})
    return trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clue and object generator")
    parser.add_argument("--grid-size", type=int, default=10, help="grid dimension")
    parser.add_argument("--trials", type=int, default=34, help="number of trials")
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data = generate_trials(args.grid_size, args.trials)
    print(json.dumps(data))
