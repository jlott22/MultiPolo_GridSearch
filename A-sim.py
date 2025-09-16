#!/usr/bin/env python3
"""
Pololu Search Simulator (v3) — A-00-matching behavior, fixed placements per episode,
batch + viewer, deterministic with seed.

Key behaviors (as requested):
- Team sizes: 1, 2, or 4 robots. 4 = four corners facing inward; 2 = opposite corners facing inward.
- Goal selection: argmax(prob_map * REWARD_FACTOR) over unknown cells, excluding any peer-reserved goals.
  Tie-break = "first to reserve" per tick (deterministic robot-id order).
- Prob map: base uniform + clue bumps 5/(1+Manhattan). Visited cells have probability 0.
  Only KNOWN (discovered) clues contribute to the probability map.
- A* on 4-connected grid with costs:
    MOVE_COST + TURN_COST(if heading changes) +
    centerward_step_cost (pre-clue only) +
    VISITED_STEP_PENALTY (onto visited) +
    INTENT_PENALTY (into a peer's current or next-intended cell)
  Node priority receives reward bonus: - REWARD_FACTOR * prob(node).
- Centerward penalty is only applied until the FIRST clue is found by the team.
- Intent handling: current-frame truth only (no TTL).
- Episode ends immediately when any robot steps on the object.
- Object placement uniform over all cells. Clues drawn WITHOUT replacement by distance kernel f(r).
  Default f(r) = 1/(1+r). Editable; also supports 1/(1+r^2).
- Modes:
    * --mode batch : run N episodes, write CSV
    * --mode view  : pygame viewer with pause/step and live FPS control via '[' and ']'

Usage examples:
  python3 A-sim.py --mode batch --episodes 200 --robots 4 --grid 10 --seed 7 --csv out.csv
  python3 A-sim.py --mode batch --robots 2 --clues 3 --clue-kernel one_over_1_plus_r2 --episodes 500
  python3 A-sim.py --mode view --robots 4 --grid 10 --seed 7 --viewer-fps 2 --show-truth
  
  J-mac:mimic simulation lottjames22$ python3 A-sim.py --mode view --robots 4 --grid 10 --seed 7 --viewer-fps 2 --show-truth --clues 2

Author: ChatGPT (for James)
"""

from __future__ import annotations
import argparse
import csv
import heapq
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ------------------------
# Constants (A-00 defaults — overridable via CLI)
# ------------------------
MOVE_COST = 1.0
TURN_COST = 1.0

DEFAULT_REWARD_FACTOR = 5.0
DEFAULT_CENTER_STEP = 0.4
DEFAULT_VISITED_STEP_PENALTY = 1.2
DEFAULT_INTENT_PENALTY = 8.0

# ------------------------
# Types & helpers
# ------------------------
Vec = Tuple[int, int]
Cell = Tuple[int, int]
DIRS4: List[Vec] = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N,E,S,W

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# ------------------------
# Config
# ------------------------
@dataclass
class Config:
    grid_size: int = 10
    robots: int = 1
    clue_count: int = 2
    clue_kernel: str = "one_over_1_plus_r"   # {"one_over_1_plus_r","one_over_1_plus_r2"}
    reward_factor: float = DEFAULT_REWARD_FACTOR
    center_step: float = DEFAULT_CENTER_STEP
    visited_step_penalty: float = DEFAULT_VISITED_STEP_PENALTY
    intent_penalty: float = DEFAULT_INTENT_PENALTY
    episodes: int = 100
    mode: str = "batch"
    seed: Optional[int] = None
    max_steps_factor: int = 2  # hard cap = grid^2 * factor (safety)
    csv_path: str = "sim_results.csv"
    viewer_fps: int = 2
    show_truth_in_viewer: bool = True

# ------------------------
# Truth world & team knowledge
# ------------------------
@dataclass
class World:
    size: int
    object_cell: Cell
    clue_cells: List[Cell]  # Fixed for the entire episode

@dataclass
class Knowledge:
    visited: Dict[Cell, int] = field(default_factory=dict)
    known_clues: List[Cell] = field(default_factory=list)
    first_clue_seen: bool = False

# ------------------------
# Placement sampling
# ------------------------
def kernel_one_over_1_plus_r(r: int) -> float:
    return 1.0 / (1 + r)

def kernel_one_over_1_plus_r2(r: int) -> float:
    return 1.0 / (1 + r*r)

KERNELS: Dict[str, Callable[[int], float]] = {
    "one_over_1_plus_r": kernel_one_over_1_plus_r,
    "one_over_1_plus_r2": kernel_one_over_1_plus_r2,
}

def sample_object_and_clues(size: int, n_clues: int, rng: random.Random,
                            kernel: Callable[[int], float]) -> Tuple[Cell, List[Cell]]:
    obj = (rng.randrange(size), rng.randrange(size))
    # All cells except object
    cells = [(x, y) for y in range(size) for x in range(size) if (x, y) != obj]
    weights = [kernel(manhattan((x, y), obj)) for (x, y) in cells]
    if all(w <= 0 for w in weights):
        weights = [1.0] * len(cells)
    s = sum(weights)
    weights = [w / s for w in weights]
    clues: List[Cell] = []
    for _ in range(n_clues):
        c = rng.choices(cells, weights=weights, k=1)[0]
        clues.append(c)
        i = cells.index(c)
        cells.pop(i); weights.pop(i)
        if not cells:
            break
        s = sum(weights)
        weights = [w / s for w in weights]
    return obj, clues

# ------------------------
# Robot agent
# ------------------------
@dataclass
class Robot:
    rid: str
    pos: Cell
    heading: Vec
    cfg: Config
    world: World
    know: Knowledge
    rng: random.Random

    current_goal: Optional[Cell] = None
    steps_taken: int = 0
    replan_count: int = 0
    path_history: List[Cell] = field(default_factory=list)

    def _idx(self, x: int, y: int) -> int:
        return y * self.cfg.grid_size + x

    def build_prob_map(self) -> List[float]:
        size = self.cfg.grid_size
        total_cells = size * size
        prob_map = [0.0] * (size * size)
        for y in range(size):
            for x in range(size):
                cell = (x, y)
                if cell in self.know.visited:
                    prob_map[self._idx(x, y)] = 0.0
                    continue
                base = 1.0 / total_cells
                clue_sum = 0.0
                for (cx, cy) in self.know.known_clues:
                    clue_sum += 5.0 / (1 + abs(x - cx) + abs(y - cy))
                prob_map[self._idx(x, y)] = base + clue_sum
        return prob_map

    def centerward_step_cost(self, curr: Cell, nxt: Cell) -> float:
        if self.know.first_clue_seen:
            return 0.0
        size = self.cfg.grid_size
        c = (size - 1) / 2.0
        cost = 0.0
        (cx, cy), (nx, ny) = curr, nxt
        if nx != cx:
            d0 = abs(cx - c); d1 = abs(nx - c)
            if d1 < d0:
                cost += self.cfg.center_step * (d0 - d1)
        if ny != cy:
            d0 = abs(cy - c); d1 = abs(ny - c)
            if d1 < d0:
                cost += self.cfg.center_step * (d0 - d1)
        return cost

    def plan_path(self, goal: Cell,
                  peer_positions: Dict[str, Cell],
                  peer_intents: Dict[str, Optional[Cell]],
                  prob_map: List[float]) -> List[Cell]:
        size = self.cfg.grid_size
        def inb(x, y): return 0 <= x < size and 0 <= y < size

        start = self.pos
        start_dir = self.heading

        frontier: List[Tuple[float, Cell, Vec]] = []
        heapq.heappush(frontier, (0.0, start, start_dir))
        came_from: Dict[Cell, Optional[Cell]] = {start: None}
        cost_so_far: Dict[Cell, float] = {start: 0.0}

        while frontier:
            _, cur, cur_dir = heapq.heappop(frontier)
            if cur == goal:
                break
            for dx, dy in DIRS4:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not inb(nx, ny):
                    continue
                nxt = (nx, ny)

                # Costs
                move = MOVE_COST
                turn = TURN_COST if (dx, dy) != cur_dir else 0.0
                visited_pen = self.cfg.visited_step_penalty if (nxt in self.know.visited) else 0.0
                serp = self.centerward_step_cost(cur, nxt)

                intent_pen = 0.0
                # into peer current
                for pid, ppos in peer_positions.items():
                    if pid != self.rid and ppos == nxt:
                        intent_pen = self.cfg.intent_penalty
                        break
                if intent_pen == 0.0:
                    # into peer intended
                    for pid, pint in peer_intents.items():
                        if pid != self.rid and pint is not None and pint == nxt:
                            intent_pen = self.cfg.intent_penalty
                            break

                step_cost = move + turn + visited_pen + serp + intent_pen
                new_cost = cost_so_far[cur] + step_cost
                if (nxt not in cost_so_far) or (new_cost < cost_so_far[nxt]):
                    cost_so_far[nxt] = new_cost
                    reward_bonus = self.cfg.reward_factor * prob_map[self._idx(nx, ny)]
                    h = manhattan(nxt, goal)
                    priority = new_cost + h - reward_bonus
                    heapq.heappush(frontier, (priority, nxt, (dx, dy)))
                    came_from[nxt] = cur

        if goal not in came_from:
            return [start]
        # reconstruct path [start, ..., goal]
        path: List[Cell] = []
        t = goal
        while t is not None:
            path.append(t)
            t = came_from[t]
        path.reverse()
        return path

    def pick_goal(self, prob_map: List[float], reserved_goals: Dict[str, Cell]) -> Optional[Cell]:
        size = self.cfg.grid_size
        reserved = set(reserved_goals.values())
        best, best_val = None, -1e30

        # Prefer the cell straight ahead when it ties with others (matches firmware).
        fx, fy = self.pos[0] + self.heading[0], self.pos[1] + self.heading[1]
        if 0 <= fx < size and 0 <= fy < size:
            forward = (fx, fy)
            if forward not in self.know.visited and forward not in reserved:
                best = forward
                best_val = prob_map[self._idx(fx, fy)] * self.cfg.reward_factor

        for y in range(size):
            for x in range(size):
                cell = (x, y)
                if cell in self.know.visited:
                    continue
                if cell in reserved:
                    continue
                val = prob_map[self._idx(x, y)] * self.cfg.reward_factor
                if val > best_val:
                    best_val = val
                    best = cell
        if best is None:
            # fallback: nearest unvisited that's not reserved
            unknown = [(x, y) for y in range(size) for x in range(size)
                       if (x, y) not in self.know.visited and (x, y) not in reserved_goals.values()]
            if not unknown:
                return None
            best = min(unknown, key=lambda c: manhattan(self.pos, c))
        return best

    def step_once(self, reserved_goals: Dict[str, Cell],
                  peer_positions: Dict[str, Cell],
                  peer_intents: Dict[str, Optional[Cell]]) -> Tuple[bool, Optional[str]]:
        # Build prob map
        prob_map = self.build_prob_map()

        # Select / reselect goal; reserve immediately (first-come via rid order)
        goal = self.pick_goal(prob_map, reserved_goals)
        if goal is None:
            return False, "no_goal"
        if goal != self.current_goal:
            self.current_goal = goal
            self.replan_count += 1
        reserved_goals[self.rid] = self.current_goal

        # Plan
        path = self.plan_path(self.current_goal, peer_positions, peer_intents, prob_map)
        if len(path) <= 1:
            peer_intents[self.rid] = None
            return False, "stuck"

        next_cell = path[1]
        peer_intents[self.rid] = next_cell

        # Move 1 step
        self.steps_taken += 1
        prev = self.pos
        self.pos = next_cell
        self.path_history.append(self.pos)
        self.heading = (self.pos[0] - prev[0], self.pos[1] - prev[1])

        # Team-shared visited
        self.know.visited[self.pos] = self.know.visited.get(self.pos, 0) + 1

        # Discover clue?
        if self.pos in self.world.clue_cells and self.pos not in self.know.known_clues:
            self.know.known_clues.append(self.pos)
            self.know.first_clue_seen = True

        # Found object?
        if self.pos == self.world.object_cell:
            return True, "found_object"

        return False, None

# ------------------------
# Start states
# ------------------------
def start_states(cfg: Config) -> list[tuple[str, tuple[int,int], tuple[int,int]]]:
    """Return list of (rid, start_pos, start_heading). Headings are (dx,dy)."""
    size = cfg.grid_size
    n = cfg.robots

    if n == 1:
        # 00: bottom-left, face North (up)
        return [("00", (0, size - 1), (0, -1))]

    if n == 2:
        return [
            # 00: bottom-left, North
            ("00", (0, size - 1), (0, -1)),
            # 01: top-right, South (down)
            ("01", (size - 1, 0), (0, 1)),
        ]

    if n == 4:
        return [
            # 00: bottom-left, North
            ("00", (0, size - 1), (0, -1)),
            # 01: top-right, South
            ("01", (size - 1, 0), (0, 1)),
            # 02: top-left, East (right)
            ("02", (0, 0), (1, 0)),
            # 03: bottom-right, West (left)
            ("03", (size - 1, size - 1), (-1, 0)),
        ]

    raise ValueError("robots must be 1, 2, or 4")


# ------------------------
# Episode runner
# ------------------------
@dataclass
class EpisodeResult:
    found: bool
    steps_total: int
    steps_per_robot: Dict[str, int]
    object_cell: Cell
    clue_cells: List[Cell]
    discovered_clues: int
    replan_counts: Dict[str, int]
    revisits: int

def run_episode(cfg: Config, rng: random.Random) -> EpisodeResult:
    # Fixed truth for the episode
    obj, clues = sample_object_and_clues(cfg.grid_size, cfg.clue_count, rng, KERNELS[cfg.clue_kernel])
    world = World(cfg.grid_size, obj, clues)
    know = Knowledge()

    # Robots
    robots: List[Robot] = []
    for rid, pos, heading in start_states(cfg):
        rb = Robot(rid=rid, pos=pos, heading=heading, cfg=cfg, world=world, know=know, rng=rng)
        rb.path_history.append(pos)
        know.visited[pos] = know.visited.get(pos, 0) + 1
        robots.append(rb)

    max_steps = cfg.grid_size * cfg.grid_size * cfg.max_steps_factor
    steps_total = 0

    # Bus state (kept current-frame)
    peer_positions: Dict[str, Cell] = {rb.rid: rb.pos for rb in robots}
    peer_intents: Dict[str, Optional[Cell]] = {rb.rid: None for rb in robots}

    found = False
    while steps_total < max_steps and not found:
        # Clear reservations each frame; "first to reserve" is per-id order this tick
        reserved_goals: Dict[str, Cell] = {}

        # (Optional safeguard) If absolutely no clue seen after ~2*grid steps, drop center penalty
        if not know.first_clue_seen and steps_total >= 2 * cfg.grid_size:
            know.first_clue_seen = True

        # Deterministic processing order by rid
        for rb in sorted(robots, key=lambda r: r.rid):
            if found:
                break
            peer_positions = {r.rid: r.pos for r in robots}
            f, _ = rb.step_once(reserved_goals, peer_positions, peer_intents)
            peer_positions[rb.rid] = rb.pos
            if f:
                found = True
                break

        steps_total += 1

    steps_per_robot = {rb.rid: rb.steps_taken for rb in robots}
    replan_counts = {rb.rid: rb.replan_count for rb in robots}
    revisits = sum(v - 1 for v in know.visited.values() if v > 1)
    return EpisodeResult(
        found=found,
        steps_total=steps_total,
        steps_per_robot=steps_per_robot,
        object_cell=world.object_cell,
        clue_cells=world.clue_cells,
        discovered_clues=len(know.known_clues),
        replan_counts=replan_counts,
        revisits=revisits,
    )

# ------------------------
# Batch mode
# ------------------------
def run_batch(cfg: Config) -> None:
    rng = random.Random(cfg.seed)
    with open(cfg.csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = [
            "episode","found","steps_total","robots","grid","object_x","object_y",
            "clues","discovered_clues","revisits","replans_total","seed"
        ] + [f"steps_{i:02d}" for i in range(cfg.robots)]
        w.writerow(cols)
        for ep in range(cfg.episodes):
            ep_seed = rng.randrange(1 << 30) if cfg.seed is not None else None
            ep_rng = random.Random(ep_seed)
            res = run_episode(cfg, ep_rng)
            replans_total = sum(res.replan_counts.values())
            row = [
                ep, int(res.found), res.steps_total, cfg.robots, cfg.grid_size,
                res.object_cell[0], res.object_cell[1],
                cfg.clue_count, res.discovered_clues, res.revisits, replans_total, ep_seed
            ] + [res.steps_per_robot.get(f"{i:02d}", 0) for i in range(cfg.robots)]
            w.writerow(row)

# ------------------------
# Viewer mode
# ------------------------
def run_viewer(cfg: Config) -> None:
    try:
        import pygame
    except Exception:
        print("pygame is required for --mode view. Install with: python3 -m pip install pygame", file=sys.stderr)
        raise

    rng = random.Random(cfg.seed)
    ep_seed = rng.randrange(1 << 30) if cfg.seed is not None else None
    ep_rng = random.Random(ep_seed)

    def new_episode():
        obj, clues = sample_object_and_clues(cfg.grid_size, cfg.clue_count, ep_rng, KERNELS[cfg.clue_kernel])
        world = World(cfg.grid_size, obj, clues)
        know = Knowledge()
        robots: List[Robot] = []
        for rid, pos, heading in start_states(cfg):
            rb = Robot(rid=rid, pos=pos, heading=heading, cfg=cfg, world=world, know=know, rng=ep_rng)
            rb.path_history.append(pos)
            know.visited[pos] = know.visited.get(pos, 0) + 1
            robots.append(rb)
        peer_positions = {rb.rid: rb.pos for rb in robots}
        peer_intents = {rb.rid: None for rb in robots}
        return world, know, robots, peer_positions, peer_intents

    world, know, robots, peer_positions, peer_intents = new_episode()

    pygame.init()
    cell_px = 40
    margin = 40
    W = cfg.grid_size * cell_px + 2 * margin
    H = cfg.grid_size * cell_px + 2 * margin + 40
    win = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    paused = False
    font = pygame.font.SysFont(None, 20)

    def draw():
        win.fill((30, 30, 35))
        # grid + visited shading
        for y in range(cfg.grid_size):
            for x in range(cfg.grid_size):
                cx = margin + x * cell_px
                cy = margin + y * cell_px
                rect = pygame.Rect(cx, cy, cell_px - 1, cell_px - 1)
                v = know.visited.get((x, y), 0)
                if v > 0:
                    shade = clamp(40 + 35 * v, 40, 200)
                    pygame.draw.rect(win, (shade, shade, 220), rect)
                else:
                    pygame.draw.rect(win, (60, 60, 70), rect, 1)

        # truth (debug) — only if requested
        if cfg.show_truth_in_viewer:
            ox, oy = world.object_cell
            pygame.draw.rect(win, (220, 60, 60),
                             pygame.Rect(margin + ox * cell_px, margin + oy * cell_px, cell_px - 1, cell_px - 1), 3)
            for (cx, cy) in world.clue_cells:
                pygame.draw.circle(win, (240, 220, 60),
                                   (margin + cx * cell_px + cell_px // 2, margin + cy * cell_px + cell_px // 2), 6, 2)

        # known clues
        for (cx, cy) in know.known_clues:
            pygame.draw.circle(win, (240, 240, 120),
                               (margin + cx * cell_px + cell_px // 2, margin + cy * cell_px + cell_px // 2), 5)

        # robots + goals
        colors = [(120, 200, 255), (255, 160, 90), (170, 255, 120), (255, 120, 190)]
        for i, rb in enumerate(sorted(robots, key=lambda r: r.rid)):
            x, y = rb.pos
            pygame.draw.rect(win, colors[i % 4],
                             pygame.Rect(margin + x * cell_px + 4, margin + y * cell_px + 4, cell_px - 8, cell_px - 8))
            if rb.current_goal:
                gx, gy = rb.current_goal
                pygame.draw.rect(win, (255, 255, 255),
                                 pygame.Rect(margin + gx * cell_px + 10, margin + gy * cell_px + 10,
                                             cell_px - 20, cell_px - 20), 1)

        txt = (
            f"robots={cfg.robots} grid={cfg.grid_size} seed={ep_seed} "
            f"paused={paused} fps={cfg.viewer_fps} known_clues={len(know.known_clues)}  "
            f"legend: robot=color block, goal=white square, clue=small dot, object=red box (--show-truth)"
        )
        t = font.render(txt, True, (230, 230, 230))
        win.blit(t, (margin, H - 30))
        pygame.display.flip()

    found = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_SPACE:
                    if paused and not found:
                        # one frame
                        reserved_goals: Dict[str, Cell] = {}
                        for rb in sorted(robots, key=lambda r: r.rid):
                            peer_positions = {r.rid: r.pos for r in robots}
                            f, _ = rb.step_once(reserved_goals, peer_positions, peer_intents)
                            if f:
                                found = True
                                break
                if event.key == pygame.K_n:
                    world, know, robots, peer_positions, peer_intents = new_episode()
                    found = False
                    paused = False
                # live speed control
                if event.key == pygame.K_LEFTBRACKET:
                    cfg.viewer_fps = max(1, cfg.viewer_fps - 1)
                if event.key == pygame.K_RIGHTBRACKET:
                    cfg.viewer_fps = min(60, cfg.viewer_fps + 1)

        if not paused and not found:
            reserved_goals = {}
            for rb in sorted(robots, key=lambda r: r.rid):
                peer_positions = {r.rid: r.pos for r in robots}
                f, _ = rb.step_once(reserved_goals, peer_positions, peer_intents)
                if f:
                    found = True
                    break

        draw()
        clock.tick(cfg.viewer_fps)

# ------------------------
# CLI
# ------------------------
def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Pololu Search Simulator v3 (A-00-matching)")
    p.add_argument("--mode", choices=["batch", "view"], default="batch")
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--robots", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--clues", type=int, default=2)
    p.add_argument("--clue-kernel", choices=list(KERNELS.keys()), default="one_over_1_plus_r")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--csv", dest="csv_path", default="sim_results.csv")
    p.add_argument("--reward-factor", type=float, default=DEFAULT_REWARD_FACTOR)
    p.add_argument("--center-step", type=float, default=DEFAULT_CENTER_STEP)
    p.add_argument("--visited-step-penalty", type=float, default=DEFAULT_VISITED_STEP_PENALTY)
    p.add_argument("--intent-penalty", type=float, default=DEFAULT_INTENT_PENALTY)
    p.add_argument("--max-steps-factor", type=int, default=2)
    p.add_argument("--viewer-fps", type=int, default=2)
    p.add_argument("--show-truth", action="store_true")
    a = p.parse_args(argv)
    return Config(
        grid_size=a.grid,
        robots=a.robots,
        clue_count=a.clues,
        clue_kernel=a.clue_kernel,
        reward_factor=a.reward_factor,
        center_step=a.center_step,
        visited_step_penalty=a.visited_step_penalty,
        intent_penalty=a.intent_penalty,
        episodes=a.episodes,
        mode=a.mode,
        seed=a.seed,
        max_steps_factor=a.max_steps_factor,
        csv_path=a.csv_path,
        viewer_fps=a.viewer_fps,
        show_truth_in_viewer=a.show_truth,
    )

def main(argv=None):
    cfg = parse_args(argv)
    if cfg.mode == "batch":
        run_batch(cfg)
        print(f"Batch complete. CSV written to: {cfg.csv_path}")
    else:
        run_viewer(cfg)

if __name__ == "__main__":
    main()
