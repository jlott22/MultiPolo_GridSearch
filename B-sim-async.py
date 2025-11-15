#!/usr/bin/env python3
# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Pololu Search Simulator (v3 async) -- B-00 forward-priority behavior with asynchronous robot clocks, fixed placements per episode,
batch + viewer.

Key behaviors (B-00 specific):
- Async scheduler: robots update on independent timelines with the same cadence as the A async model (moves ~1.50-1.70s apart, turns ~0.3s).
- Goal selection: forward/side bias before the first clue, then global argmax search; no goal reservation broadcast.
- Prob map: pre-clue uniform; post-clue applies 1/(1+Manhattan) bumps from discovered clues; visited cells stay at 0.
- Pathfinding: 4-connected A* with turn costs (halved post-clue), visited penalty, and centerward bias until first clue.
- Collision avoidance: avoids peer positions and declared intents; relies on delayed peer telemetry from the async comm bus.
- Communication: position/visited/clue/object updates ride on a configurable delay; no goal reservation channel.
- Episode ends as soon as any robot reaches the object. Trials load from generator or CSV/JSON inputs.
- Modes:
    * --mode batch : run N episodes, write CSV
    * --mode view  : pygame viewer with pause/step and live FPS control via '[' and ']'

Usage examples:
  python3 B-sim-async.py --mode batch --episodes 200 --robots 4 --grid 10 --csv out.csv
  python3 B-sim-async.py --mode batch --robots 2 --clues 3 --episodes 500 --trials-out trials.json
  python3 B-sim-async.py --mode view --robots 4 --grid 10 --viewer-fps 1 --show-truth --trials-in trials.json

Author: James Lott
TODO: fix the travel replan issue 
"""
# Viewer controls quick reference:
#  Esc/q: quit | p: pause/resume | space: step while paused
#  . and ,: move forward/back in history | [: slow playback | ]: speed playback
#  n: next trial | r: replay current trial | h/v/b/c: heatmap, visited, prob nums, contours
#  i: revisits | t: peek truth | Shift+t: toggle truth overlay | g/s/o/e: ghost, trails, paths, contrib
#  j/k/l: export JSON, CSV history, CSV summary | Home/End: jump to start or latest frame
#
# Viewer mode workflow:
#  1. Run: py B-sim-async.py --mode view
#  2. Watch Trial 0 complete (robot finds object)
#  3. Press 'n' (lowercase) to advance to Trial 1
#  4. Watch Trial 1 complete
#  5. Press 'q' or 'ESC' to quit
#  6. CSV results auto-save to: sim_results_B_async.csv (in current directory)

from __future__ import annotations
import argparse
import csv
import heapq
import json
import math
import copy
import sys
import os
import importlib.util
import datetime
import random
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from collections import deque

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ------------------------
# Constants (B-00 defaults â€” overridable via CLI)
# ------------------------
MOVE_COST = 1.0
TURN_COST = 1.0

DEFAULT_REWARD_FACTOR = 5.0
DEFAULT_CENTER_STEP = 0.4
DEFAULT_VISITED_STEP_PENALTY = 4.0
DEFAULT_INTENT_PENALTY = 8.0
DEFAULT_INTENT_SYNC_DELAY = 0.05

# ------------------------
# Types & helpers
# ------------------------
Vec = Tuple[int, int]
Cell = Tuple[int, int]
DIRS4: List[Vec] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N,E,S,W (y increases upward)
DIR_TO_INDEX: Dict[Vec, int] = {d: i for i, d in enumerate(DIRS4)}

def quarter_turns(from_dir: Optional[Vec], to_dir: Vec) -> int:
    if from_dir == to_dir:
        return 0
    if from_dir is None:
        return 1
    try:
        fi = DIR_TO_INDEX[from_dir]
        ti = DIR_TO_INDEX[to_dir]
    except KeyError:
        return 1
    delta = (ti - fi) % len(DIRS4)
    return 2 if delta == 2 else 1

NORTH, EAST, SOUTH, WEST = DIRS4

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# ------------------------
# Config
# ------------------------
@dataclass
class Config:
    grid_size: int = 30
    robots: int = 2
    clue_count: int = 2
    reward_factor: float = DEFAULT_REWARD_FACTOR
    center_step: float = DEFAULT_CENTER_STEP
    visited_step_penalty: float = DEFAULT_VISITED_STEP_PENALTY
    intent_penalty: float = DEFAULT_INTENT_PENALTY
    episodes: int = 50
    mode: str = "batch"  # "batch" or "view"
    csv_path: str = "sim_results_B1.csv"
    viewer_fps: int = 4
    cell_px: int = 15
    show_truth_in_viewer: bool = True
    trials_in: Optional[str] = None
    trials_out: Optional[str] = None
    scenarios_csv: Optional[str] = "trials_d1_c2_g30.csv"
    start_trial: int = 0  # Which trial to start with (0, 1, 2, etc). Auto-increments after each viewer run.
    async_step_mean: float = 1.6
    async_step_jitter: float = 0.05
    async_min_delay: float = 1.55
    async_max_delay: float = 1.65
    async_initial_spread: float = 1.0
    async_tick_span: float = 0.25
    async_turn_quarter: float = 0.3
    async_seed: Optional[int] = None
    intent_sync_delay: float = DEFAULT_INTENT_SYNC_DELAY
    comm_delay_s: float = .04
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
    individual_revisits: int = 0  # Tracks repeat visits by this robot (like repeat_intersection_count in A-00)
    system_revisits: int = 0  # Tracks all system-wide repeat visits (like system_repeat_count in A-00)
    first_clue_step: Optional[int] = None  # Total steps when first clue was found
    robot_positions_at_first_clue: Dict[str, Cell] = field(default_factory=dict)  # Robot positions when first clue found


def aggregate_team_knowledge(robots: List['Robot']) -> Knowledge:
    """Combine per-robot knowledge into a viewer-only team summary."""
    agg = Knowledge()
    visited: Dict[Cell, int] = {}
    known_clues: List[Cell] = []
    known_clue_set: Set[Cell] = set()
    individual_revisits = 0
    system_revisits = 0

    first_clue_seen = False
    first_clue_step: Optional[int] = None
    robot_positions_at_first_clue: Dict[str, Cell] = {}

    for rb in robots:
        for cell, count in rb.know.visited.items():
            prev = visited.get(cell, 0)
            if count > prev:
                visited[cell] = count

        for cell in rb.know.known_clues:
            if cell not in known_clue_set:
                known_clue_set.add(cell)
                known_clues.append(cell)

        individual_revisits += rb.know.individual_revisits
        system_revisits += rb.know.system_revisits

        if rb.know.first_clue_seen:
            candidate_step = rb.know.first_clue_step
            if not first_clue_seen:
                first_clue_seen = True
                first_clue_step = candidate_step
                robot_positions_at_first_clue = dict(rb.know.robot_positions_at_first_clue)
            elif candidate_step is not None and (first_clue_step is None or candidate_step < first_clue_step):
                first_clue_step = candidate_step
                robot_positions_at_first_clue = dict(rb.know.robot_positions_at_first_clue)

    agg.visited = visited
    agg.known_clues = known_clues
    agg.individual_revisits = individual_revisits
    agg.system_revisits = system_revisits
    agg.first_clue_seen = first_clue_seen
    agg.first_clue_step = first_clue_step
    agg.robot_positions_at_first_clue = robot_positions_at_first_clue
    return agg

# ------------------------
# Communication (async bus)
# ------------------------

@dataclass
class CommMessage:
    sender: str
    topic: str
    payload: str
    deliver_at: float


class CommBus:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.pending: List[CommMessage] = []
        self.modules: Dict[str, 'CommModule'] = {}

    def register(self, module: 'CommModule') -> None:
        self.modules[module.rid] = module

    def send(self, sender: str, topic: str, payload: str, now: float) -> None:
        deliver_at = now + self.delay_s
        self.pending.append(CommMessage(sender=sender, topic=topic, payload=payload, deliver_at=deliver_at))

    def pump(self, now: float) -> None:
        if not self.pending:
            return
        ready: List[CommMessage] = [msg for msg in self.pending if msg.deliver_at <= now + 1e-9]
        if not ready:
            return
        ready.sort(key=lambda msg: msg.deliver_at)
        self.pending = [msg for msg in self.pending if msg.deliver_at > now + 1e-9]
        for msg in ready:
            for rid, module in self.modules.items():
                if rid == msg.sender:
                    continue
                module.receive(msg)


class CommModule:
    def __init__(self, robot: 'Robot', bus: CommBus) -> None:
        self.robot = robot
        self.rid = robot.rid
        self.bus = bus
        self.current_time = 0.0
        self.peer_positions: Dict[str, Cell] = {}
        self.peer_intents: Dict[str, Optional[Cell]] = {}
        self.peer_goals: Dict[str, Optional[Cell]] = {}
        self.system_visits: Dict[Cell, int] = {}
        self.received_clues: Set[Cell] = set()
        self.object_location: Optional[Cell] = None
        self.start_signal: bool = True

        self.position_msgs_sent = 0
        self.position_msgs_received = 0
        self.visited_msgs_sent = 0
        self.visited_msgs_received = 0
        self.clue_msgs_sent = 0
        self.clue_msgs_received = 0
        self.object_msgs_sent = 0
        self.object_msgs_received = 0
        self.intent_msgs_sent = 0
        self.intent_msgs_received = 0
        self.goal_msgs_sent = 0
        self.goal_msgs_received = 0

        self._last_sent_position: Optional[Cell] = None
        self._last_sent_goal: Optional[Cell] = None
        self._last_sent_intent: Optional[Cell] = None
        self._object_announced = False
        self._sent_visited: Set[Cell] = set()
        self._sent_clues: Set[Cell] = set()

        self.bus.register(self)

    def set_time(self, now: float) -> None:
        self.current_time = now

    def _send(self, topic: str, payload: str) -> None:
        self.bus.send(self.rid, topic, payload, self.current_time)

    def publish_position(self, cell: Cell) -> None:
        if self._last_sent_position == cell:
            return
        self._last_sent_position = cell
        self.position_msgs_sent += 1
        self._send('1', f"{cell[0]},{cell[1]}")

    def publish_visited(self, cell: Cell) -> None:
        if cell in self._sent_visited:
            return
        self._sent_visited.add(cell)
        self.visited_msgs_sent += 1
        self._send('2', f"{cell[0]},{cell[1]}")

    def publish_clue(self, cell: Cell) -> None:
        if cell in self._sent_clues:
            return
        self._sent_clues.add(cell)
        self.clue_msgs_sent += 1
        self._send('3', f"{cell[0]},{cell[1]}")

    def publish_object(self, cell: Cell) -> None:
        if self._object_announced:
            return
        self._object_announced = True
        self.object_msgs_sent += 1
        self._send('4', f"{cell[0]},{cell[1]}")

    def publish_intent(self, cell: Optional[Cell]) -> None:
        if cell is None:
            self.clear_intent()
            return
        if self._last_sent_intent == cell:
            return
        self._last_sent_intent = cell
        self.intent_msgs_sent += 1
        self._send('5', f"{cell[0]},{cell[1]}")

    def clear_intent(self) -> None:
        self._last_sent_intent = None

    def publish_goal(self, cell: Optional[Cell]) -> None:
        if cell is None:
            self.clear_goal()
            return
        if self._last_sent_goal == cell:
            return
        self._last_sent_goal = cell
        self.goal_msgs_sent += 1
        self._send('7', f"{cell[0]},{cell[1]}")

    def clear_goal(self) -> None:
        self._last_sent_goal = None

    def _parse_cell(self, payload: str) -> Optional[Cell]:
        try:
            x_str, y_str = payload.split(',', 1)
            return int(x_str), int(y_str)
        except ValueError:
            return None

    def receive(self, msg: CommMessage) -> None:
        payload = msg.payload.strip()
        if msg.topic == '1':
            cell = self._parse_cell(payload)
            if cell is None:
                return
            self.peer_positions[msg.sender] = cell
            self.position_msgs_received += 1
        elif msg.topic == '2':
            cell = self._parse_cell(payload)
            if cell is None:
                return
            # Just track that we've seen this cell visited by another robot
            # (system_revisits are counted in step_once when robot actually visits)
            self.system_visits[cell] = self.system_visits.get(cell, 0) + 1
            self.visited_msgs_received += 1
        elif msg.topic == '3':
            cell = self._parse_cell(payload)
            if cell is None:
                return
            self.received_clues.add(cell)
            self.clue_msgs_received += 1
        elif msg.topic == '4':
            cell = self._parse_cell(payload)
            if cell is None:
                return
            self.object_location = cell
            self.object_msgs_received += 1
        elif msg.topic == '5':
            cell = self._parse_cell(payload)
            if cell is None:
                return
            self.peer_intents[msg.sender] = cell
            self.intent_msgs_received += 1
        elif msg.topic == '6':
            self.start_signal = payload == '1'
        elif msg.topic == '7':
            cell = self._parse_cell(payload)
            if cell is None:
                return
            self.peer_goals[msg.sender] = cell
            self.goal_msgs_received += 1

    def snapshot_peer_positions(self) -> Dict[str, Cell]:
        return {rid: pos for rid, pos in self.peer_positions.items() if rid != self.rid}

    def snapshot_peer_intents(self) -> Dict[str, Optional[Cell]]:
        return {rid: cell for rid, cell in self.peer_intents.items() if rid != self.rid}

    def snapshot_peer_goals(self) -> Dict[str, Optional[Cell]]:
        return {rid: cell for rid, cell in self.peer_goals.items() if rid != self.rid}
# ------------------------
# Trials (external generator only)
# ------------------------

def _load_external_generator():
    here = os.path.dirname(__file__)
    gen_path = os.path.join(here, "clue_object_generator_manhat.py")
    if not os.path.isfile(gen_path):
        raise RuntimeError("External generator file 'clue_object_generator_manhat.py' not found; the simulator requires it.")
    spec = importlib.util.spec_from_file_location("clue_object_generator_manhat", gen_path)
    if not spec or not spec.loader:
        raise RuntimeError("Failed to load external generator module spec.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def generate_trial_list(cfg: Config) -> List[Dict[str, List[Cell]]]:
    """Generate trials using the external generator's ordering.

    Calls `generate_trials(grid_size, num_trials, clues_per_object)` from the external
    file so the ordering and mode schedule are exactly as that file defines.
    """
    mod = _load_external_generator()
    trials = mod.generate_trials(cfg.grid_size, cfg.episodes, cfg.clue_count)
    return trials

def load_trial_list(path: str) -> List[Dict[str, List[Cell]]]:
    with open(path, "r") as f:
        data = json.load(f)
    # Expect list of {"object": [x,y], "clues": [[x,y], ...]}
    trials: List[Dict[str, List[Cell]]] = []
    for item in data:
        obj = tuple(item["object"])  # type: ignore
        clues = [tuple(c) for c in item["clues"]]
        trials.append({"object": obj, "clues": clues})
    return trials

def load_scenarios_csv(path: str) -> List[Dict[str, List[Cell]]]:
    """Load trial scenarios from CSV file generated by clue_object_generator."""
    trials: List[Dict[str, List[Cell]]] = []

    with open(path, "r", newline='') as csvfile:
        # Skip comment lines starting with #
        lines = [line for line in csvfile if not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            # Parse object location
            obj = (int(row['object_x']), int(row['object_y']))

            # Parse clue locations (handle variable number of clues)
            clues = []
            clue_num = 1
            while f'clue{clue_num}_x' in row and row[f'clue{clue_num}_x']:
                clue_x = int(row[f'clue{clue_num}_x'])
                clue_y = int(row[f'clue{clue_num}_y'])
                clues.append((clue_x, clue_y))
                clue_num += 1

            trials.append({"object": obj, "clues": clues})

    return trials

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

    current_goal: Optional[Cell] = None
    steps_taken: int = 0
    steps_after_first_clue: int = 0  # Steps taken after first clue found
    system_revisits: int = 0  # This robot's revisits to cells visited by ANY robot
    collision_replans: int = 0
    goal_replans: int = 0
    path_replan_pending: bool = False
    goal_replan_pending: bool = False
    goal_completed_by_self_pending: bool = False
    path_history: List[Cell] = field(default_factory=list)
    last_path: List[Cell] = field(default_factory=list, repr=False)
    last_next_cell: Optional[Cell] = None
    last_step_reason: Optional[str] = None
    last_event: Optional[str] = None
    last_prob_map: List[float] = field(default_factory=list, repr=False)
    comm: Optional['CommModule'] = field(default=None, repr=False)
    pending_actions: Deque[Dict[str, Any]] = field(default_factory=deque, repr=False)
    next_action_time: float = 0.0
    _perception_current_positions: Dict[str, Cell] = field(default_factory=dict, repr=False)
    _perception_current_intents: Dict[str, Optional[Cell]] = field(default_factory=dict, repr=False)
    _perception_current_goals: Dict[str, Optional[Cell]] = field(default_factory=dict, repr=False)
    _perception_pending_positions: Dict[str, Cell] = field(default_factory=dict, repr=False)
    _perception_pending_intents: Dict[str, Optional[Cell]] = field(default_factory=dict, repr=False)
    _perception_pending_goals: Dict[str, Optional[Cell]] = field(default_factory=dict, repr=False)
    _perception_pending_valid: bool = False
    _perception_initialized: bool = False

    position_msgs_sent: int = 0
    position_msgs_received: int = 0
    visited_msgs_sent: int = 0
    visited_msgs_received: int = 0
    clue_msgs_sent: int = 0
    clue_msgs_received: int = 0
    object_msgs_sent: int = 0
    object_msgs_received: int = 0
    intent_msgs_sent: int = 0
    intent_msgs_received: int = 0

    def _idx(self, x: int, y: int) -> int:
        return y * self.cfg.grid_size + x

    def sync_from_comm(self) -> None:
        """
        Apply any delayed peer information that has been delivered via the comm module.
        Mirrors the firmware, where knowledge only advances once a UART frame arrives.
        """
        if self.comm is None:
            return

        for cell, count in self.comm.system_visits.items():
            existing = self.know.visited.get(cell, 0)
            if count > existing:
                self.know.visited[cell] = count

        new_clue = False
        for cell in self.comm.received_clues:
            if cell not in self.know.known_clues:
                self.know.known_clues.append(cell)
                new_clue = True

        if new_clue and not self.know.first_clue_seen:
            self.know.first_clue_seen = True

    def _capture_perception(self) -> None:
        """Record the peer view for the next planning cycle."""
        if self.comm is None:
            self._perception_pending_positions = {}
            self._perception_pending_intents = {}
            self._perception_pending_goals = {}
        else:
            self.sync_from_comm()
            self._perception_pending_positions = self.comm.snapshot_peer_positions()
            self._perception_pending_intents = self.comm.snapshot_peer_intents()
            self._perception_pending_goals = self.comm.snapshot_peer_goals()
        self._perception_pending_valid = True

    def _promote_perception(self) -> Tuple[Dict[str, Cell], Dict[str, Optional[Cell]], Dict[str, Optional[Cell]]]:
        """Return the frozen peer data used for planning, promoting a new snapshot if ready."""
        if not self._perception_initialized and not self._perception_pending_valid:
            self._capture_perception()
        if self._perception_pending_valid:
            self._perception_current_positions = dict(self._perception_pending_positions)
            self._perception_current_intents = dict(self._perception_pending_intents)
            self._perception_current_goals = dict(self._perception_pending_goals)
            self._perception_pending_valid = False
            self._perception_initialized = True
        return (
            self._perception_current_positions,
            self._perception_current_intents,
            self._perception_current_goals,
        )

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
                if len(self.know.known_clues) == 0:
                    prob_map[self._idx(x, y)] = 1.0 / total_cells
                else:
                    acc = 0.0
                    for (cx, cy) in self.know.known_clues:
                        d = abs(x - cx) + abs(y - cy)
                        acc += 1.0 / (1.0 + d)
                    prob_map[self._idx(x, y)] = acc
        return prob_map

    def centerward_step_cost(self, curr: Cell, nxt: Cell) -> float:
        if self.know.first_clue_seen:
            return 0.0
        size = self.cfg.grid_size
        c = (size - 1) / 2.0
        cost = 0.0
        (cx, cy), (nx, ny) = curr, nxt
        if nx != cx:
            d0 = abs(cx - c)
            d1 = abs(nx - c)
            if d1 < d0:
                cost += self.cfg.center_step * (d0 - d1)
        if ny != cy:
            d0 = abs(cy - c)
            d1 = abs(ny - c)
            if d1 < d0:
                cost += self.cfg.center_step * (d0 - d1)
        return cost

    def plan_path(self, goal: Cell, peer_positions: Dict[str, Cell], peer_intents: Dict[str, Optional[Cell]], prob_map: List[float]) -> List[Cell]:
        size = self.cfg.grid_size

        def inb(x: int, y: int) -> bool:
            return 0 <= x < size and 0 <= y < size

        start = self.pos
        start_dir = self.heading
        turn_cost_per_turn = TURN_COST if not self.know.first_clue_seen else TURN_COST * 0.5

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

                move = MOVE_COST
                turns = quarter_turns(cur_dir, (dx, dy))
                turn = turn_cost_per_turn * turns
                visited_pen = self.cfg.visited_step_penalty if (nxt in self.know.visited) else 0.0
                serp = self.centerward_step_cost(cur, nxt)

                blocked_by_peer = any(
                    pid != self.rid and ppos == nxt
                    for pid, ppos in peer_positions.items()
                )
                if not blocked_by_peer:
                    blocked_by_peer = any(
                        pid != self.rid and pint is not None and pint == nxt
                        for pid, pint in peer_intents.items()
                    )
                if blocked_by_peer:
                    continue

                reward_bonus = self.cfg.reward_factor * prob_map[self._idx(nx, ny)]
                base_cost = move + turn + visited_pen + serp
                max_bonus = max(0.0, base_cost - 0.01)
                if reward_bonus > max_bonus:
                    reward_bonus = max_bonus
                step_cost = base_cost - reward_bonus
                if step_cost < 0.01:
                    step_cost = 0.01

                new_cost = cost_so_far[cur] + step_cost
                if (nxt not in cost_so_far) or (new_cost < cost_so_far[nxt]):
                    cost_so_far[nxt] = new_cost
                    h = manhattan(nxt, goal)
                    heapq.heappush(frontier, (new_cost + h, nxt, (dx, dy)))
                    came_from[nxt] = cur

        if goal not in came_from:
            return [start]

        path: List[Cell] = []
        t = goal
        while t is not None:
            path.append(t)
            t = came_from[t]
        path.reverse()
        return path

    def pick_goal(self, prob_map: List[float]) -> Tuple[Optional[Cell], float]:
        size = self.cfg.grid_size
        best: Optional[Cell] = None
        best_val = float('-inf')

        fx = self.pos[0] + self.heading[0]
        fy = self.pos[1] + self.heading[1]
        if self.heading != (0, 0) and 0 <= fx < size and 0 <= fy < size:
            forward = (fx, fy)
            if forward not in self.know.visited:
                idx = self._idx(fx, fy)
                best = forward
                best_val = prob_map[idx] * self.cfg.reward_factor

        if best is None and self.heading != (0, 0):
            left = (-self.heading[1], self.heading[0])
            right = (self.heading[1], -self.heading[0])
            for sx, sy in (left, right):
                sx_cell = (self.pos[0] + sx, self.pos[1] + sy)
                if not (0 <= sx_cell[0] < size and 0 <= sx_cell[1] < size):
                    continue
                if sx_cell in self.know.visited:
                    continue
                idx = self._idx(sx_cell[0], sx_cell[1])
                best = sx_cell
                best_val = prob_map[idx] * self.cfg.reward_factor
                break

        for y in range(size):
            for x in range(size):
                cell = (x, y)
                if cell in self.know.visited:
                    continue
                val = prob_map[self._idx(x, y)] * self.cfg.reward_factor
                if val > best_val:
                    best_val = val
                    best = cell

        if best is None:
            unknown = [
                (x, y)
                for y in range(size)
                for x in range(size)
                if (x, y) not in self.know.visited
            ]
            if not unknown:
                return None, float('-inf')
            best = min(unknown, key=lambda c: manhattan(self.pos, c))
            best_val = prob_map[self._idx(best[0], best[1])] * self.cfg.reward_factor
        return best, best_val

    def simple_energy_metrics(self, elapsed_ms: int) -> Dict[str, Any]:
        if self.comm is not None:
            self.position_msgs_sent = self.comm.position_msgs_sent
            self.position_msgs_received = self.comm.position_msgs_received
            self.visited_msgs_sent = self.comm.visited_msgs_sent
            self.visited_msgs_received = self.comm.visited_msgs_received
            self.clue_msgs_sent = self.comm.clue_msgs_sent
            self.clue_msgs_received = self.comm.clue_msgs_received
            self.object_msgs_sent = self.comm.object_msgs_sent
            self.object_msgs_received = self.comm.object_msgs_received
            self.intent_msgs_sent = self.comm.intent_msgs_sent
            self.intent_msgs_received = self.comm.intent_msgs_received
        compute_time_ms = elapsed_ms
        total_msgs_sent = (self.position_msgs_sent + self.visited_msgs_sent + self.clue_msgs_sent +
                           self.object_msgs_sent + self.intent_msgs_sent)
        total_msgs_received = (self.position_msgs_received + self.visited_msgs_received +
                               self.clue_msgs_received + self.object_msgs_received + self.intent_msgs_received)
        return {
            'compute_time_ms': compute_time_ms,
            'msgs_sent': total_msgs_sent,
            'msgs_received': total_msgs_received,
        }

    def preview_goal(self, reserved_goals: Optional[Dict[str, Cell]] = None) -> None:
        if self.comm is None:
            raise RuntimeError("preview_goal requires an active CommModule for coordination.")
        self.sync_from_comm()
        prob_map = self.build_prob_map()
        goal, _ = self.pick_goal(prob_map)
        self.current_goal = goal
        if reserved_goals is not None:
            reserved_goals[self.rid] = goal

    def step_once(self, reserved_goals: Dict[str, Cell], peer_positions: Dict[str, Cell], peer_intents: Dict[str, Optional[Cell]]) -> Tuple[bool, Optional[str]]:
        if self.comm is None:
            raise RuntimeError("Robot.step_once requires an active CommModule for coordination.")

        if self.pending_actions:
            return self._execute_pending_action(reserved_goals, peer_positions, peer_intents)

        plan_peer_positions, plan_peer_intents, _plan_peer_goals = self._promote_perception()
        prob_map = self.build_prob_map()
        self.last_prob_map = prob_map[:]
        self.last_event = None
        self.last_step_reason = None

        previous_goal = self.current_goal
        if self.current_goal is not None:
            if self.current_goal in self.know.visited:
                if self.pos == self.current_goal:
                    self.goal_completed_by_self_pending = True
                else:
                    self.goal_completed_by_self_pending = False
                self.goal_replan_pending = True
                self.current_goal = None
                self._clear_pending_actions()
                self.last_next_cell = None

        current_goal_reward = float('-inf')
        if self.current_goal is not None:
            idx = self._idx(self.current_goal[0], self.current_goal[1])
            current_goal_reward = prob_map[idx] * self.cfg.reward_factor

        candidate_goal, candidate_reward = self.pick_goal(prob_map)
        goal = candidate_goal
        eps = 1e-9
        if self.current_goal is not None:
            if goal is None:
                goal = self.current_goal
                candidate_reward = current_goal_reward
            elif goal != self.current_goal and candidate_reward <= current_goal_reward + eps:
                goal = self.current_goal
                candidate_reward = current_goal_reward

        if goal is None:
            self.current_goal = None
            self.last_path = []
            self.last_next_cell = None
            self.last_step_reason = "no_goal"
            self.last_event = "no_goal"
            self._clear_pending_actions()
            if self.comm:
                self.comm.clear_intent()
            reserved_goals[self.rid] = None
            self._capture_perception()
            return False, "no_goal"

        goal_changed = goal != previous_goal
        self.current_goal = goal
        if goal_changed:
            self.last_event = "goal_changed"
            # After the first clue, every actual goal change counts as a replan
            if self.know.first_clue_seen:
                self.goal_replans += 1
            self.goal_replan_pending = False
            self.goal_completed_by_self_pending = False

        path = self.plan_path(self.current_goal, plan_peer_positions, plan_peer_intents, prob_map)
        self.last_path = path[:]
        if len(path) <= 1:
            peer_intents[self.rid] = None
            reserved_goals[self.rid] = self.current_goal
            if self.comm:
                self.comm.clear_intent()
            self.last_next_cell = None
            self.last_step_reason = "stuck"
            if self.last_event is None:
                self.last_event = "stuck"
            self._clear_pending_actions()
            self._capture_perception()
            return False, "stuck"

        next_cell = path[1]
        self._queue_actions_for_next_cell(
            next_cell,
            plan_peer_positions,
            plan_peer_intents,
        )
        reserved_goals[self.rid] = self.current_goal
        self._consume_path_replan()
        return self._execute_pending_action(reserved_goals, peer_positions, peer_intents)

    def _clear_pending_actions(self) -> None:
        self.pending_actions.clear()

    def _consume_path_replan(self) -> None:
        if self.path_replan_pending:
            self.collision_replans += 1
            self.path_replan_pending = False

    def _queue_actions_for_next_cell(
        self,
        next_cell: Cell,
        peer_positions: Optional[Dict[str, Cell]] = None,
        peer_intents: Optional[Dict[str, Optional[Cell]]] = None,
    ) -> None:
        self._clear_pending_actions()
        move_vec = (next_cell[0] - self.pos[0], next_cell[1] - self.pos[1])
        if move_vec not in DIRS4:
            raise ValueError(f"Next cell {next_cell} not adjacent to {self.pos}")
        current_heading = self.heading if self.heading in DIRS4 else (0, 0)
        desired_heading = move_vec
        if current_heading == (0, 0) and desired_heading != (0, 0):
            self.pending_actions.append({
                "kind": "turn",
                "heading": desired_heading,
                "target": next_cell,
            })
        elif current_heading != desired_heading:
            cur_idx = DIRS4.index(current_heading)
            desired_idx = DIRS4.index(desired_heading)
            cw_steps = (desired_idx - cur_idx) % len(DIRS4)
            ccw_steps = (cur_idx - desired_idx) % len(DIRS4)
            if cw_steps <= ccw_steps:
                step = 1
                steps = cw_steps
            else:
                step = -1
                steps = ccw_steps
            for _ in range(steps):
                cur_idx = (cur_idx + step) % len(DIRS4)
                new_heading = DIRS4[cur_idx]
                self.pending_actions.append({
                    "kind": "turn",
                    "heading": new_heading,
                    "target": next_cell,
                })
        self.pending_actions.append({
            "kind": "move",
            "heading": desired_heading,
            "target": next_cell,
            "peer_positions": dict(peer_positions) if peer_positions is not None else None,
            "peer_intents": dict(peer_intents) if peer_intents is not None else None,
        })
        self.last_next_cell = next_cell

    def _execute_pending_action(self, reserved_goals: Dict[str, Cell], peer_positions: Dict[str, Cell], peer_intents: Dict[str, Optional[Cell]]) -> Tuple[bool, Optional[str]]:
        if self.comm is None:
            raise RuntimeError("_execute_pending_action requires an active CommModule for coordination.")

        if not self.pending_actions:
            peer_intents[self.rid] = None
            reserved_goals[self.rid] = self.current_goal
            self.comm.clear_intent()
            return False, None

        action = self.pending_actions.popleft()
        kind = action.get("kind")
        target = action.get("target")
        heading: Vec = action.get("heading", self.heading)
        ground_peer_positions = {pid: pos for pid, pos in peer_positions.items() if pid != self.rid}

        if self.current_goal is not None:
            reserved_goals[self.rid] = self.current_goal
        else:
            reserved_goals[self.rid] = None

        if target is not None:
            peer_intents[self.rid] = target
            self.last_next_cell = target
            if not action.get('intent_announced'):
                self.comm.publish_intent(target)
                action['intent_announced'] = True
        else:
            peer_intents[self.rid] = None
            self.comm.clear_intent()
            if not self.pending_actions:
                self.last_next_cell = None

        if kind == "turn":
            self.heading = heading
            if self.last_event is None:
                self.last_event = "turn"
            self.last_step_reason = "turn"
            return False, "turn"

        if kind == "move":
            next_cell = target
            if self.cfg.intent_sync_delay > 0 and not action.get('intent_sync_delay_applied'):
                action['intent_sync_delay_applied'] = True
                self.pending_actions.appendleft(action)
                self.last_event = "intent_sync"
                self.last_step_reason = "intent_sync"
                return False, "intent_sync"
            if next_cell is None:
                self.last_event = "stuck"
                self.last_step_reason = "stuck"
                self._capture_perception()
                self._clear_pending_actions()
                self.last_next_cell = None
                self.comm.clear_intent()
                return False, "stuck"

            # Use delayed comm data for collision detection (matches real robot behavior)
            # Real robots only know peer state from delayed UART messages
            blocked = False
            stored_intents = action.get("peer_intents")
            perceived_intents = stored_intents if stored_intents is not None else self.comm.snapshot_peer_intents()
            stored_positions = action.get("peer_positions")
            perceived_positions = stored_positions if stored_positions is not None else self.comm.snapshot_peer_positions()

            for pid, pint in perceived_intents.items():
                if pint is not None and pint == next_cell:
                    blocked = True
                    break
            if not blocked:
                for pid, pos in perceived_positions.items():
                    if pos == next_cell:
                        blocked = True
                        break
            if not blocked:
                for pid, pos in ground_peer_positions.items():
                    if pos == next_cell:
                        blocked = True
                        break

            if blocked:
                self.path_replan_pending = True
                self.last_event = "collision_yield"
                self.last_step_reason = "collision_yield"
                self._capture_perception()
                self._clear_pending_actions()
                self.last_next_cell = None
                if self.comm:
                    self.comm.clear_intent()
                return False, "collision_yield"
            self.steps_taken += 1
            if self.know.first_clue_seen:
                self.steps_after_first_clue += 1
            self.pos = next_cell
            self.path_history.append(self.pos)
            self.heading = heading
            if self.comm:
                self.comm.publish_position(self.pos)

            # Track system revisits BEFORE publishing (when this robot visits a cell ANY robot has visited)
            if self.comm and self.pos in self.comm.system_visits:
                self.system_revisits += 1
                self.know.system_revisits += 1

            # Track individual revisits (matches A-00.py record_intersection behavior)
            if self.pos in self.know.visited:
                self.know.individual_revisits += 1
            self.know.visited[self.pos] = self.know.visited.get(self.pos, 0) + 1

            # Update system_visits with this robot's visit
            if self.comm:
                self.comm.system_visits[self.pos] = self.comm.system_visits.get(self.pos, 0) + 1
                self.comm.publish_visited(self.pos)
            if self.pos in self.world.clue_cells and self.pos not in self.know.known_clues:
                self.know.known_clues.append(self.pos)
                was_first_clue = not self.know.first_clue_seen
                self.know.first_clue_seen = True
                if self.comm:
                    self.comm.publish_clue(self.pos)
                self.last_event = "clue_found"
            if self.pos == self.world.object_cell:
                self.last_step_reason = "found_object"
                if self.last_event != "clue_found":
                    self.last_event = "found_object"
                if self.comm:
                    self.comm.publish_object(self.pos)
                self._capture_perception()
                self._clear_pending_actions()
                peer_intents[self.rid] = None
                if self.comm:
                    self.comm.clear_intent()
                self.last_next_cell = None
                return True, "found_object"
            if self.last_event is None:
                self.last_event = "moved"
            self.last_step_reason = self.last_event
            self._capture_perception()
            self._clear_pending_actions()
            peer_intents[self.rid] = None
            if self.comm:
                self.comm.clear_intent()
            self.last_next_cell = None
            return False, None

        self._clear_pending_actions()
        self.last_step_reason = "stuck"
        if self.last_event is None:
            self.last_event = "stuck"
        self.last_next_cell = None
        if self.comm:
            self.comm.clear_intent()
        return False, "stuck"

# ------------------------
# Start states
# ------------------------
def start_states(cfg: Config) -> list[tuple[str, tuple[int,int], tuple[int,int]]]:
    """Return list of (rid, start_pos, start_heading). Headings are (dx,dy)."""
    size = cfg.grid_size
    n = cfg.robots


    # With (0,0) as bottom-left, y increases upward
    if n == 1:
        # 00: bottom-left, face North (up)
        return [("00", (0, 0), NORTH)]

    if n == 2:
        return [
            # 00: bottom-left, North
            ("00", (0, 0), NORTH),
            # 01: top-right, South (down)
            ("01", (size - 1, size - 1), SOUTH),
        ]

    if n == 4:
        return [
            # 00: bottom-left, North
            ("00", (0, 0), NORTH),
            # 01: top-right, South
            ("01", (size - 1, size - 1), SOUTH),
            # 02: top-left, East
            ("02", (0, size - 1), EAST),
            # 03: bottom-right, West
            ("03", (size - 1, 0), WEST),
        ]

    raise ValueError("robots must be 1, 2, or 4")


# ------------------------
# Episode runner
# ------------------------
@dataclass
class EpisodeResult:
    found: bool
    steps_total: int
    object_cell: Cell
    clue_cells: List[Cell]
    discovered_clues: int
    discovered_clue_locations: List[Cell]  # Clues discovered, in order
    steps_before_first_clue: int
    steps_after_first_clue: int
    robot_positions_at_first_clue: Dict[str, Cell]  # Robot positions when first clue found
    collision_replans: Dict[str, int]
    goal_replans: Dict[str, int]
    collision_replans_total: int
    goal_replans_total: int
    revisits: int
    system_revisits: int
    unique_cells: int  # Total unique cells visited by the team

    @property
    def replan_counts(self) -> Dict[str, int]:
        # Keep for backwards compatibility if needed
        robot_ids = set(self.collision_replans) | set(self.goal_replans)
        return {rid: self.collision_replans.get(rid, 0) + self.goal_replans.get(rid, 0) for rid in robot_ids}

def collect_episode_result(world: World, know: Knowledge, robots: List['Robot'], steps_total: int, found: bool) -> EpisodeResult:
    collision_replans = {rb.rid: rb.collision_replans for rb in robots}
    goal_replans = {rb.rid: rb.goal_replans for rb in robots}
    collision_replans_total = sum(collision_replans.values())
    goal_replans_total = sum(goal_replans.values())

    # Calculate steps before/after first clue using team totals
    if know.first_clue_step is not None:
        steps_before_first_clue = know.first_clue_step
        steps_after_first_clue = steps_total - know.first_clue_step
    else:
        steps_before_first_clue = steps_total
        steps_after_first_clue = 0

    return EpisodeResult(
        found=found,
        steps_total=steps_total,
        object_cell=world.object_cell,
        clue_cells=world.clue_cells,
        discovered_clue_locations=know.known_clues,
        steps_before_first_clue=steps_before_first_clue,
        steps_after_first_clue=steps_after_first_clue,
        robot_positions_at_first_clue=know.robot_positions_at_first_clue,
        collision_replans=collision_replans,
        goal_replans=goal_replans,
        collision_replans_total=collision_replans_total,
        goal_replans_total=goal_replans_total,
        discovered_clues=len(know.known_clues),
        revisits=know.individual_revisits,
        system_revisits=know.system_revisits,
        unique_cells=len(know.visited),
    )

@dataclass
class AsyncEvent:
    robot: Robot
    time: float
    found_object: bool
    reason: Optional[str]


class AsyncScheduler:
    def __init__(self, cfg: Config, robots: List[Robot], rng: Optional[random.Random] = None, comm_bus: Optional[CommBus] = None):
        self.cfg = cfg
        self.robots = robots
        self.rng = rng or random.Random()
        self.clock = 0.0
        self.robot_map = {rb.rid: rb for rb in robots}
        if comm_bus is None:
            raise ValueError("AsyncScheduler requires a CommBus instance")
        self.comm_bus = comm_bus
        self.reserved_goals: Dict[str, Optional[Cell]] = {rb.rid: rb.current_goal for rb in robots}
        self.peer_intents: Dict[str, Optional[Cell]] = {rb.rid: None for rb in robots}
        self.next_action: Dict[str, float] = {}
        base_interval = max(self.cfg.async_step_mean, self.cfg.async_min_delay)
        spread = max(0.0, self.cfg.async_initial_spread)
        span = base_interval * spread if spread > 0 else base_interval
        for rb in robots:
            offset = self.rng.uniform(0.0, span) if span > 0 else self.rng.uniform(0.0, base_interval)
            self.next_action[rb.rid] = offset
            rb.next_action_time = offset

    def _move_interval(self) -> float:
        jitter = self.rng.uniform(-self.cfg.async_step_jitter, self.cfg.async_step_jitter)
        interval = self.cfg.async_step_mean + jitter
        interval = max(self.cfg.async_min_delay, interval)
        if self.cfg.async_max_delay > 0:
            interval = min(interval, self.cfg.async_max_delay)
        return max(interval, 1e-3)

    def _interval_for_reason(self, reason: Optional[str]) -> float:
        if reason == "turn":
            return max(self.cfg.async_turn_quarter, 1e-3)
        if reason == "intent_sync":
            return max(self.cfg.intent_sync_delay, 1e-3)
        if reason == "collision_yield":
            # Real robot waits 300ms when yielding to peer, then replans
            return 0.3
        return self._move_interval()

    def step_until(self, time_limit: float, allow_overshoot: bool = True) -> List[AsyncEvent]:
        events: List[AsyncEvent] = []
        progressed = False
        while self.next_action:
            rid, next_time = min(self.next_action.items(), key=lambda item: item[1])
            if next_time > time_limit:
                if progressed or not allow_overshoot:
                    break
                time_limit = next_time
            self.clock = next_time
            self.comm_bus.pump(self.clock)
            robot = self.robot_map[rid]
            if robot.comm:
                robot.comm.set_time(self.clock)
            peer_positions = {r.rid: r.pos for r in self.robots}

            # Check if this will be the first clue
            was_first_clue_before = robot.know.first_clue_seen
            found_object, reason = robot.step_once(self.reserved_goals, peer_positions, self.peer_intents)

            # If first clue was just found, capture robot positions and total steps
            if not was_first_clue_before and robot.know.first_clue_seen and robot.know.first_clue_step is None:
                robot.know.first_clue_step = sum(r.steps_taken for r in self.robots)
                robot.know.robot_positions_at_first_clue = {r.rid: r.pos for r in self.robots}

            self.reserved_goals[rid] = robot.current_goal
            if rid not in self.peer_intents:
                self.peer_intents[rid] = None
            events.append(AsyncEvent(robot=robot, time=self.clock, found_object=found_object, reason=reason))
            progressed = True
            interval = self._interval_for_reason(reason)
            self.next_action[rid] = next_time + interval
            robot.next_action_time = self.next_action[rid]
            if found_object:
                break
        return events

    def next_wakeup(self) -> Optional[float]:
        if not self.next_action:
            return None
        return min(self.next_action.values())


def make_async_rng(cfg: Config, episode_index: Optional[int] = None) -> random.Random:
    if cfg.async_seed is None:
        return random.Random()
    seed = cfg.async_seed if episode_index is None else cfg.async_seed + int(episode_index) * 9973
    return random.Random(seed)

class MetricsPlotter:
    def __init__(self) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required for MetricsPlotter")
        try:
            import matplotlib
            # Try different backends in order of preference
            backends = ['Qt5Agg', 'TkAgg', 'Agg']
            for backend in backends:
                try:
                    matplotlib.use(backend, force=True)
                    break
                except (ImportError, Exception):
                    continue
        except Exception:
            pass  # Use whatever backend is available
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            try:
                manager.set_window_title("Episode Metrics")
            except Exception:
                pass
        self.ax.axis("off")
        self._last_episode_index: Optional[int] = None
        self._last_metrics: Optional[EpisodeResult] = None

    def update(self, metrics: EpisodeResult, cfg: Config, episode_index: Optional[int], tick: int) -> None:
        self._last_metrics = metrics
        self._last_episode_index = episode_index
        self.ax.clear()
        self.ax.axis("off")
        collision_total = sum(metrics.collision_replans.values())
        goal_total = sum(metrics.goal_replans.values())
        replans_total = collision_total + goal_total
        base_lines = [
            f"episode={episode_index if episode_index is not None else '-'}",
            f"tick={tick}",
            f"found={int(metrics.found)}",
            f"steps_total={metrics.steps_total}",
            f"steps_before={metrics.steps_before_first_clue} steps_after={metrics.steps_after_first_clue}",
            f"robots={cfg.robots} grid={cfg.grid_size}",
            f"object=({metrics.object_cell[0]}, {metrics.object_cell[1]})",
            f"clues={cfg.clue_count} discovered={metrics.discovered_clues}",
            f"revisits={metrics.revisits} system_revisits={metrics.system_revisits}",
            f"collision_total={collision_total} goal_total={goal_total} replans_total={replans_total}",
        ]
        for idx, line in enumerate(base_lines):
            self.ax.text(0.02, 0.95 - idx * 0.07, line, transform=self.ax.transAxes, ha="left", va="top", fontsize=11, family="monospace")
        robot_ids = sorted(set(metrics.collision_replans) | set(metrics.goal_replans))
        robot_lines = []
        for rid in robot_ids:
            collision = metrics.collision_replans.get(rid, 0)
            goal = metrics.goal_replans.get(rid, 0)
            robot_lines.append(f"{rid}: goal_replans={goal} collision={collision}")
        if not robot_lines:
            robot_lines.append("no robot replan data")
        for idx, line in enumerate(robot_lines):
            self.ax.text(0.52, 0.95 - idx * 0.07, line, transform=self.ax.transAxes, ha="left", va="top", fontsize=11, family="monospace")

        csv_values = [
            episode_index if episode_index is not None else 0,
            int(metrics.found),
            metrics.steps_total,
            metrics.steps_before_first_clue,
            metrics.steps_after_first_clue,
            cfg.robots,
            cfg.grid_size,
            metrics.object_cell[0],
            metrics.object_cell[1],
            cfg.clue_count,
            metrics.discovered_clues,
            metrics.revisits,
            metrics.system_revisits,
            collision_total,
            goal_total,
            replans_total,
        ]
        csv_text = ", ".join(str(value) for value in csv_values)
        self.ax.text(0.02, 0.05, f"csv_row: {csv_text}", transform=self.ax.transAxes, ha="left", va="bottom", fontsize=9, family="monospace")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self._pump()

    def _pump(self) -> None:
        if plt is None:
            return
        plt.pause(0.001)

    def pump_events(self) -> None:
        if plt is None:
            return
        plt.pause(0.001)


def run_episode(cfg: Config, trial: Tuple[Cell, List[Cell]], episode_index: Optional[int] = None) -> EpisodeResult:
    obj, clues = trial
    world = World(cfg.grid_size, obj, clues)
    comm_bus = CommBus(cfg.comm_delay_s)

    robots: List[Robot] = []
    found_at_spawn = False
    for rid, pos, heading in start_states(cfg):
        rb_knowledge = Knowledge()
        rb = Robot(rid=rid, pos=pos, heading=heading, cfg=cfg, world=world, know=rb_knowledge)
        rb.path_history.append(pos)
        rb.know.visited[pos] = rb.know.visited.get(pos, 0) + 1
        if rb.pos in world.clue_cells and rb.pos not in rb.know.known_clues:
            rb.know.known_clues.append(rb.pos)
            rb.know.first_clue_seen = True
            if rb.know.first_clue_step is None:
                rb.know.first_clue_step = 0
                rb.know.robot_positions_at_first_clue = {rb.rid: rb.pos}
        if rb.pos == world.object_cell:
            found_at_spawn = True
        rb.comm = CommModule(rb, comm_bus)
        rb.comm.set_time(0.0)
        rb.comm.publish_position(rb.pos)
        rb.comm.publish_visited(rb.pos)
        if rb.pos in world.clue_cells:
            rb.comm.publish_clue(rb.pos)
        if rb.pos == world.object_cell:
            rb.comm.publish_object(rb.pos)

        robots.append(rb)

    team_know = aggregate_team_knowledge(robots)
    if found_at_spawn:
        steps_total = sum(rb.steps_taken for rb in robots)
        return collect_episode_result(world, team_know, robots, steps_total, True)

    rng = make_async_rng(cfg, episode_index)
    scheduler = AsyncScheduler(cfg, robots, rng, comm_bus)
    tick_span = cfg.async_tick_span if cfg.async_tick_span > 0 else max(cfg.async_step_mean, cfg.async_max_delay, cfg.async_min_delay, cfg.async_turn_quarter)
    safety_limit = max(10, cfg.grid_size * cfg.grid_size * 20)
    ticks = 0
    found = False
    while not found and ticks < safety_limit:
        events = scheduler.step_until(scheduler.clock + tick_span, allow_overshoot=True)
        if not events:
            break
        ticks += 1
        if any(ev.found_object for ev in events):
            found = True
            break
    steps_total = sum(rb.steps_taken for rb in robots)
    team_know = aggregate_team_knowledge(robots)
    return collect_episode_result(world, team_know, robots, steps_total, found)

# ------------------------
# Batch mode
# ------------------------
def run_batch(cfg: Config) -> None:
    # Prepare trial list
    if cfg.scenarios_csv:
        trials = load_scenarios_csv(cfg.scenarios_csv)
    elif cfg.trials_in:
        trials = load_trial_list(cfg.trials_in)
    else:
        trials = generate_trial_list(cfg)
        if cfg.trials_out:
            # write JSON-friendly lists
            with open(cfg.trials_out, "w") as tf:
                json.dump(trials, tf)

    with open(cfg.csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # New column specification
        cols = [
            "episode", "grid", "robots", "clues", "steps_total",
            "steps_before_first_clue", "steps_after_first_clue",
            "object_x", "object_y", "clue_locations", "found",
            "discovered_clue_locations", "robot_positions_at_first_clue",
            "collision_replans_total", "goal_replans_total",
            "revisits", "system_revisits", "unique_cells"
        ]

        w.writerow(cols)
        total_eps = min(cfg.episodes, len(trials)) if (cfg.trials_in or cfg.scenarios_csv) else cfg.episodes
        for ep in range(total_eps):
            trial = (tuple(trials[ep]["object"]), [tuple(c) for c in trials[ep]["clues"]])
            res = run_episode(cfg, trial, episode_index=ep)

            # Format clue locations as string
            clue_locs_str = ";".join(f"({c[0]},{c[1]})" for c in res.clue_cells)
            discovered_locs_str = ";".join(f"({c[0]},{c[1]})" for c in res.discovered_clue_locations)
            robot_pos_str = ";".join(f"{rid}:({pos[0]},{pos[1]})" for rid, pos in sorted(res.robot_positions_at_first_clue.items()))

            row = [
                ep, cfg.grid_size, cfg.robots, cfg.clue_count, res.steps_total,
                res.steps_before_first_clue, res.steps_after_first_clue,
                res.object_cell[0], res.object_cell[1], clue_locs_str, int(res.found),
                discovered_locs_str, robot_pos_str,
                res.collision_replans_total, res.goal_replans_total,
                res.revisits, res.system_revisits, res.unique_cells
            ]

            w.writerow(row)

# ------------------------
# Viewer mode
# ------------------------


def run_viewer(cfg: Config) -> None:
    try:
        import pygame
        pygame.init()
    except Exception:
        print("pygame is required for --mode view. Install with: python3 -m pip install pygame", file=sys.stderr)
        raise

    # Load trial counter from file if it exists
    counter_file = "viewer_trial_counter_B_async.txt"
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r") as f:
                cfg.start_trial = int(f.read().strip())
        except:
            pass  # Use default if file is corrupted

    if cfg.scenarios_csv:
        trials = load_scenarios_csv(cfg.scenarios_csv)
    elif cfg.trials_in:
        trials = load_trial_list(cfg.trials_in)
    else:
        trials = generate_trial_list(cfg)
        if cfg.trials_out:
            with open(cfg.trials_out, "w") as tf:
                json.dump(trials, tf)

    class ViewerApp:
        def __init__(self, cfg: Config, trials: List[Dict[str, List[Cell]]]):
            self.cfg = cfg
            self.trials = trials
            self.trial_index = cfg.start_trial
            self.last_trial_index: Optional[int] = None
            self.active_trial_index: Optional[int] = None
            self.viewer_fps = max(0.25, float(cfg.viewer_fps))
            self.min_fps = 0.25
            self.max_fps = 60.0
            self.max_history = 5000
            self.event_log: List[Dict[str, Any]] = []
            self.history: List[Dict[str, Any]] = []
            self.history_view_index = -1
            self.follow_history = True
            self.async_scheduler: Optional[AsyncScheduler] = None
            self.async_tick_span = 0.0
            self.async_rng: Optional[random.Random] = None
            self.comm_bus: Optional[CommBus] = None
            self.collision_highlights: List[Dict[str, Any]] = []
            self.collision_cells_for_snapshot: List[Cell] = []
            self.selected_robot: Optional[str] = None
            self.toggles = {
                "show_heatmap": True,
                "show_visited": True,
                "show_prob_numbers": True,
                "show_revisits": True,
                "contour": False,
                "show_truth": cfg.show_truth_in_viewer,
                "peek_truth": False,
                "show_contrib": False,
                "show_trails": True,
                "show_paths": True,
                "show_ghost": True,
            }
            self.metrics_plotter: Optional[MetricsPlotter] = None
            if plt is not None:
                try:
                    self.metrics_plotter = MetricsPlotter()
                except Exception as exc:
                    exc_msg = str(exc).replace('\n', ' ')[:100] + "..." if len(str(exc)) > 100 else str(exc)
                    print(f"Warning: metrics plotter disabled ({exc_msg})", file=sys.stderr)
                    self.metrics_plotter = None
            else:
                self.metrics_plotter = None
            self.tick_count = 0
            self.steps_since_clue = 0
            self.found = False
            self.found_by: Optional[str] = None
            self.found_tick: Optional[int] = None
            self.ghost_path: List[Cell] = []
            self.initial_positions: Dict[str, Cell] = {}
            self.world: World
            self.know: Knowledge
            self.robots: List[Robot] = []
            self.viewer_accumulator = 0.0
            self.paused = False
            self.cell_px = 40
            self.margin = 40
            self.panel_w = 260
            self.info_h = 140
            self.grid_px = cfg.grid_size * self.cell_px
            width = self.grid_px + 2 * self.margin + self.panel_w
            height = self.grid_px + 2 * self.margin + self.info_h
            self.win = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Pololu Search Viewer (async)")
            self.font = pygame.font.SysFont(None, 20)
            self.font_small = pygame.font.SysFont(None, 16)
            self.font_micro = pygame.font.SysFont(None, 14)
            self.robot_palette = [
                (120, 200, 255),
                (255, 160, 90),
                (170, 255, 120),
                (255, 120, 190),
            ]
            self.export_counter = 0
            self.completed_trials: List[Dict[str, Any]] = []
            self.new_episode(reset_trial_index=False)
        def world_to_screen(self, cell: Cell, size: Optional[int] = None) -> Tuple[int, int]:
            if size is None:
                size = self.cfg.grid_size
            x, y = cell
            sx = self.margin + x * self.cell_px
            sy = self.margin + (size - 1 - y) * self.cell_px
            return sx, sy

        def world_to_screen_center(self, cell: Cell, size: Optional[int] = None) -> Tuple[int, int]:
            sx, sy = self.world_to_screen(cell, size)
            return sx + self.cell_px // 2, sy + self.cell_px // 2

        def screen_to_world(self, px: int, py: int, size: Optional[int] = None) -> Optional[Cell]:
            if size is None:
                size = self.cfg.grid_size
            if not (self.margin <= px < self.margin + self.grid_px and self.margin <= py < self.margin + self.grid_px):
                return None
            cell_x = (px - self.margin) // self.cell_px
            cell_y_from_top = (py - self.margin) // self.cell_px
            cell_y = size - 1 - cell_y_from_top
            if not (0 <= cell_x < size and 0 <= cell_y < size):
                return None
            return (cell_x, cell_y)
        def log_event(self, kind: str, message: str, robot: Optional[str] = None,
                      cell: Optional[Cell] = None, extra: Optional[Dict[str, Any]] = None) -> None:
            entry: Dict[str, Any] = {
                "tick": self.tick_count,
                "kind": kind,
                "message": message,
            }
            if robot is not None:
                entry["robot"] = robot
            if cell is not None:
                entry["cell"] = [cell[0], cell[1]]
            if extra:
                entry.update(extra)
            self.event_log.append(entry)
            if len(self.event_log) > 400:
                self.event_log = self.event_log[-400:]

        def serialize_robot(self, rb: Robot) -> Dict[str, Any]:
            if not rb.last_prob_map:
                rb.last_prob_map = rb.build_prob_map()
            return {
                "rid": rb.rid,
                "pos": [rb.pos[0], rb.pos[1]],
                "current_goal": [rb.current_goal[0], rb.current_goal[1]] if rb.current_goal else None,
                "steps_taken": rb.steps_taken,
                "collision_replans": rb.collision_replans,
                "goal_replans": rb.goal_replans,
                "path_history": [[x, y] for (x, y) in rb.path_history[-200:]],
                "last_path": [[x, y] for (x, y) in rb.last_path],
                "last_next_cell": [rb.last_next_cell[0], rb.last_next_cell[1]] if rb.last_next_cell else None,
                "last_step_reason": rb.last_step_reason,
                "last_event": rb.last_event,
                "last_prob_map": rb.last_prob_map[:],
                "next_action_time": rb.next_action_time,
            }

        def serialize_state(self, note: str) -> Dict[str, Any]:
            visited_payload = [
                {
                    "cell": [cell[0], cell[1]],
                    "count": count,
                }
                for cell, count in self.know.visited.items()
            ]
            snapshot: Dict[str, Any] = {
                "note": note,
                "tick": self.tick_count,
                "steps_since_clue": self.steps_since_clue,
                "found": self.found,
                "found_by": self.found_by,
                "sim_clock": self.async_scheduler.clock if self.async_scheduler else 0.0,
                "world": {
                    "size": self.world.size,
                    "object": [self.world.object_cell[0], self.world.object_cell[1]],
                    "clues": [[c[0], c[1]] for c in self.world.clue_cells],
                },
                "knowledge": {
                    "visited": visited_payload,
                    "known_clues": [[c[0], c[1]] for c in self.know.known_clues],
                    "first_clue_seen": self.know.first_clue_seen,
                },
                "robots": [self.serialize_robot(rb) for rb in self.robots],
                "collisions": [[cell[0], cell[1]] for cell in self.collision_cells_for_snapshot],
                "ghost_path": [[c[0], c[1]] for c in self.ghost_path],
            }
            return snapshot

        def capture_snapshot(self, note: str) -> None:
            snapshot = self.serialize_state(note)
            self.history.append(snapshot)
            if len(self.history) > self.max_history:
                overflow = len(self.history) - self.max_history
                self.history = self.history[overflow:]
                if self.history_view_index >= 0:
                    self.history_view_index = max(0, self.history_view_index - overflow)
            if self.follow_history or self.history_view_index == len(self.history) - 2:
                self.history_view_index = len(self.history) - 1
                self.follow_history = True
            else:
                self.history_view_index = min(self.history_view_index, len(self.history) - 1)
            self.collision_cells_for_snapshot = []

        def update_metrics_display(self) -> None:
            if not self.metrics_plotter:
                return
            if not self.robots:
                return
            self.update_team_knowledge()
            try:
                total_steps = sum(rb.steps_taken for rb in self.robots)
                metrics = collect_episode_result(self.world, self.know, self.robots, total_steps, self.found)
                self.metrics_plotter.update(metrics, self.cfg, self.active_trial_index, self.tick_count)
            except Exception as exc:
                print(f"Warning: metrics plotter disabled ({exc})", file=sys.stderr)
                self.metrics_plotter = None

        def update_team_knowledge(self) -> None:
            if not hasattr(self, "know") or self.know is None:
                self.know = Knowledge()
            aggregated = aggregate_team_knowledge(self.robots)
            self.know.visited = aggregated.visited
            self.know.known_clues = aggregated.known_clues
            self.know.individual_revisits = aggregated.individual_revisits
            self.know.system_revisits = aggregated.system_revisits
            self.know.first_clue_seen = aggregated.first_clue_seen
            self.know.first_clue_step = aggregated.first_clue_step
            self.know.robot_positions_at_first_clue = aggregated.robot_positions_at_first_clue

        def new_episode(self, reset_trial_index: bool = False, replay: bool = False) -> None:
            # Display results from previous episode if it was running
            if hasattr(self, 'robots') and self.robots and not reset_trial_index:
                self.display_episode_results()

            if reset_trial_index:
                self.trial_index = 0
                self.last_trial_index = None
            if replay and self.last_trial_index is not None:
                idx = self.last_trial_index
            else:
                idx = self.trial_index % len(self.trials)
                self.last_trial_index = idx
                self.trial_index += 1
            self.active_trial_index = idx
            if not self.trials:
                raise RuntimeError("No trials available for viewer.")
            trial = self.trials[idx]
            obj = tuple(trial["object"])
            clues = [tuple(c) for c in trial["clues"]]
            self.world = World(self.cfg.grid_size, obj, clues)
            self.know = Knowledge()
            self.robots = []
            self.found = False
            self.found_by = None
            self.found_tick = None
            self.async_scheduler = None
            self.async_rng = None
            self.comm_bus = CommBus(self.cfg.comm_delay_s)
            for rid, pos, heading in start_states(self.cfg):
                rb_knowledge = Knowledge()
                rb = Robot(rid=rid, pos=pos, heading=heading, cfg=self.cfg, world=self.world, know=rb_knowledge)
                rb.path_history.append(pos)
                rb.know.visited[pos] = rb.know.visited.get(pos, 0) + 1
                if rb.pos in self.world.clue_cells and rb.pos not in rb.know.known_clues:
                    rb.know.known_clues.append(rb.pos)
                    rb.know.first_clue_seen = True
                    if rb.know.first_clue_step is None:
                        rb.know.first_clue_step = 0
                        rb.know.robot_positions_at_first_clue = {rb.rid: rb.pos}
                if rb.pos == self.world.object_cell:
                    self.found = True
                    self.found_by = rb.rid
                    self.found_tick = 0
                rb.comm = CommModule(rb, self.comm_bus)
                rb.comm.set_time(0.0)
                rb.comm.publish_position(rb.pos)
                rb.comm.publish_visited(rb.pos)
                if rb.pos in self.world.clue_cells:
                    rb.comm.publish_clue(rb.pos)
                if rb.pos == self.world.object_cell:
                    rb.comm.publish_object(rb.pos)

                rb.preview_goal()
                rb.last_prob_map = rb.build_prob_map()
                self.robots.append(rb)
            self.update_team_knowledge()
            seed_index = self.active_trial_index if self.active_trial_index is not None else None
            self.async_rng = make_async_rng(self.cfg, seed_index)
            self.async_scheduler = AsyncScheduler(self.cfg, self.robots, self.async_rng, self.comm_bus)
            base_tick_span = self.cfg.async_tick_span if self.cfg.async_tick_span > 0 else max(self.cfg.async_step_mean, self.cfg.async_max_delay, self.cfg.async_min_delay, self.cfg.async_turn_quarter)
            if base_tick_span <= 0:
                base_tick_span = max(0.1, self.cfg.async_step_mean if self.cfg.async_step_mean > 0 else 0.1)
            self.async_tick_span = base_tick_span
            self.initial_positions = {rb.rid: rb.pos for rb in self.robots}
            self.tick_count = 0
            self.steps_since_clue = 0
            if not self.found:
                self.found_by = None
                self.found_tick = None
            self.ghost_path = []
            self.viewer_accumulator = 0.0
            self.event_log.clear()
            self.collision_highlights.clear()
            self.collision_cells_for_snapshot = []
            self.history = []
            self.history_view_index = -1
            self.follow_history = True
            self.paused = False
            self.capture_snapshot("episode_start")
            self.log_event("episode", f"Episode start: object {self.world.object_cell}", None, self.world.object_cell)

            # Log any spawn discoveries
            for rb in self.robots:
                if rb.pos in self.know.known_clues:
                    self.log_event("clue", f"{rb.rid} found clue at spawn {rb.pos}", rb.rid, rb.pos)
                if self.found and self.found_by == rb.rid:
                    self.log_event("object", f"{rb.rid} found object at spawn {rb.pos}", rb.rid, rb.pos)

            self.update_metrics_display()

        def compute_ghost_path(self) -> None:
            if self.found_by is None:
                self.ghost_path = []
                return
            start = self.initial_positions.get(self.found_by)
            if start is None:
                self.ghost_path = []
                return
            target = self.world.object_cell
            if start == target:
                self.ghost_path = [[start[0], start[1]]]
                return
            frontier = deque([start])
            came: Dict[Cell, Optional[Cell]] = {start: None}
            size = self.world.size
            while frontier:
                cur = frontier.popleft()
                if cur == target:
                    break
                for dx, dy in DIRS4:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    if not (0 <= nx < size and 0 <= ny < size):
                        continue
                    nxt = (nx, ny)
                    if nxt in came:
                        continue
                    came[nxt] = cur
                    frontier.append(nxt)
            if target not in came:
                self.ghost_path = []
                return
            path_cells: List[Cell] = []
            cur: Optional[Cell] = target
            while cur is not None:
                path_cells.append(cur)
                cur = came.get(cur)
            path_cells.reverse()
            self.ghost_path = [[c[0], c[1]] for c in path_cells]

        def advance_tick(self) -> None:
            if self.found or not self.async_scheduler:
                return
            if self.async_tick_span <= 0:
                base_span = max(self.cfg.async_step_mean, self.cfg.async_max_delay, self.cfg.async_min_delay)
                if base_span <= 0:
                    base_span = max(0.1, self.cfg.async_step_mean if self.cfg.async_step_mean > 0 else 0.1)
                self.async_tick_span = base_span
            events = self.async_scheduler.step_until(self.async_scheduler.clock + self.async_tick_span, allow_overshoot=True)
            if not events:
                return
            tick_clue_found = False
            tick_found_object = False
            new_collision_cells: List[Cell] = []
            for event in events:
                rb = event.robot
                if rb.last_event == "goal_changed" and rb.current_goal is not None:
                    self.log_event("goal", f"{rb.rid} reserved {rb.current_goal}", rb.rid, rb.current_goal)
                if rb.last_event == "clue_found":
                    tick_clue_found = True
                    self.log_event("clue", f"{rb.rid} found clue at {rb.pos}", rb.rid, rb.pos)
                if event.reason == "stuck":
                    self.log_event("stuck", f"{rb.rid} stuck at {rb.pos}", rb.rid, rb.pos)
                if event.reason == "no_goal":
                    self.log_event("nog", f"{rb.rid} had no available goal", rb.rid, rb.pos)
                if event.found_object:
                    tick_found_object = True
                    self.found = True
                    self.found_by = rb.rid
                    self.found_tick = self.tick_count + 1
                    self.log_event("object", f"{rb.rid} found object at {rb.pos}", rb.rid, rb.pos)
                    break
            intent_targets: Dict[Cell, List[str]] = {}
            for rid, cell in self.async_scheduler.peer_intents.items():
                if cell is None:
                    continue
                intent_targets.setdefault(cell, []).append(rid)
            for cell, rids in intent_targets.items():
                if len(rids) > 1:
                    self.collision_highlights.append({"cell": cell, "ttl": 1.5})
                    new_collision_cells.append(cell)
                    self.log_event("intent", f"Intent collision at {cell} by {', '.join(rids)}", None, cell, {"robots": rids})
            goal_targets: Dict[Cell, List[str]] = {}
            for rid, cell in self.async_scheduler.reserved_goals.items():
                if cell is None:
                    continue
                goal_targets.setdefault(cell, []).append(rid)
            for cell, rids in goal_targets.items():
                if len(rids) > 1 and cell not in new_collision_cells:
                    self.collision_highlights.append({"cell": cell, "ttl": 1.5})
                    new_collision_cells.append(cell)
                    self.log_event("goal_conflict", f"Goal conflict at {cell} by {', '.join(rids)}", None, cell, {"robots": rids})
            self.update_team_knowledge()
            self.tick_count += 1
            if tick_clue_found:
                self.steps_since_clue = 0
            else:
                self.steps_since_clue += 1
            if tick_found_object:
                self.compute_ghost_path()
                self.display_episode_results()
            self.collision_cells_for_snapshot = new_collision_cells
            self.capture_snapshot("tick")
            self.update_metrics_display()
        def update_highlights(self, dt: float) -> None:
            if not self.collision_highlights:
                return
            keep: List[Dict[str, Any]] = []
            for item in self.collision_highlights:
                ttl = max(0.0, item.get("ttl", 0.0) - dt)
                if ttl > 0.0:
                    keep.append({"cell": item["cell"], "ttl": ttl})
            self.collision_highlights = keep

        def current_snapshot(self) -> Dict[str, Any]:
            if not self.history:
                return self.serialize_state("empty")
            self.history_view_index = max(0, min(self.history_view_index, len(self.history) - 1))
            return self.history[self.history_view_index]

        def viewing_latest(self) -> bool:
            return self.history_view_index == len(self.history) - 1

        def adjust_history(self, delta: int) -> None:
            if not self.history:
                return
            self.history_view_index = int(clamp(self.history_view_index + delta, 0, len(self.history) - 1))
            self.follow_history = self.viewing_latest()

        def jump_to_latest(self) -> None:
            if not self.history:
                return
            self.history_view_index = len(self.history) - 1
            self.follow_history = True

        def compute_contributions(self, snapshot: Dict[str, Any]) -> Dict[Cell, List[float]]:
            size = snapshot["world"]["size"]
            visited = {
                (item["cell"][0], item["cell"][1]): item["count"]
                for item in snapshot["knowledge"]["visited"]
            }
            contributions: Dict[Cell, List[float]] = {}
            for raw in snapshot["knowledge"]["known_clues"]:
                clue = (raw[0], raw[1])
                arr = [0.0] * (size * size)
                for y in range(size):
                    for x in range(size):
                        if (x, y) in visited:
                            continue
                        d = abs(clue[0] - x) + abs(clue[1] - y)
                        arr[y * size + x] = 1.0 / (1.0 + d)
                contributions[clue] = arr
            return contributions

        def draw_event_log(self, surface, snapshot: Dict[str, Any], panel_x: int, panel_y: int) -> int:
            target_tick = snapshot["tick"]
            entries = [e for e in self.event_log if e["tick"] <= target_tick]
            entries = entries[-8:]
            y = panel_y
            for entry in entries[::-1]:
                prefix = f"[{entry['tick']:04d}] "
                msg = entry.get("message", "")
                text = prefix + msg
                label = self.font_small.render(text, True, (220, 220, 230))
                surface.blit(label, (panel_x, y))
                y += label.get_height() + 4
            return y

        def draw_sidebar(self, snapshot: Dict[str, Any]) -> None:
            panel_x = self.margin + self.grid_px + 16
            panel_y = self.margin
            panel_width = self.panel_w - 32
            panel_rect = pygame.Rect(panel_x - 12, panel_y - 12, panel_width + 24, self.grid_px + self.info_h - 16)
            pygame.draw.rect(self.win, (40, 40, 48), panel_rect, border_radius=8)
            pygame.draw.rect(self.win, (70, 70, 90), panel_rect, 1, border_radius=8)
            text_color = (235, 235, 245)
            sim_clock = snapshot.get('sim_clock', 0.0)
            info_lines = [
                f"tick={snapshot['tick']}",
                f"sim_clock={sim_clock:.2f}s",
                f"since_clue={self.steps_since_clue}",
                f"fps={self.viewer_fps:.1f}",
                f"history={'live' if self.viewing_latest() else 'scrub'}",
            ]
            if self.found_by:
                info_lines.append(f"found_by={self.found_by} at tick {self.found_tick}")
            for line in info_lines:
                label = self.font.render(line, True, text_color)
                self.win.blit(label, (panel_x, panel_y))
                panel_y += label.get_height() + 4
            panel_y += 6
            robots = snapshot["robots"]
            for idx, robot in enumerate(robots):
                rid = robot["rid"]
                color = self.robot_palette[idx % len(self.robot_palette)]
                active_color = color if self.selected_robot in (None, rid) else tuple(int(c * 0.5) for c in color)

                # Highlight robot if it's the next to act (closest next_action_time to sim_clock)
                next_action = robot.get("next_action_time", 0.0)
                time_until_action = next_action - sim_clock
                is_active = abs(time_until_action) < 0.01  # Robot just acted or is about to act

                header_text = f"Robot {rid}"
                if is_active:
                    header_text += " [ACTIVE]"
                header = self.font.render(header_text, True, active_color)
                self.win.blit(header, (panel_x, panel_y))
                panel_y += header.get_height()
                pos = tuple(robot["pos"])
                goal = tuple(robot["current_goal"]) if robot["current_goal"] else None
                stats_lines = [
                    f"pos={pos}",
                    f"goal={goal if goal else 'None'}",
                    f"next_act={next_action:.2f}s (Δ{time_until_action:+.2f}s)",
                    f"steps={robot['steps_taken']}",
                    f"goal_replans={robot['goal_replans']} collision={robot['collision_replans']}",
                    f"last={robot['last_event'] or '-'}",
                ]
                for sline in stats_lines:
                    lbl = self.font_small.render(sline, True, (210, 210, 220))
                    self.win.blit(lbl, (panel_x, panel_y))
                    panel_y += lbl.get_height()
                panel_y += 6
            panel_y += 4
            label = self.font.render("Events", True, text_color)
            self.win.blit(label, (panel_x, panel_y))
            panel_y += label.get_height() + 4
            panel_y = self.draw_event_log(self.win, snapshot, panel_x, panel_y)
            panel_y += 6
            hints = [
                "p pause, space step, ,/. scrub",
                "[ ] speed, n next, r replay",
                "h heatmap, v visited, b nums, c contour",
                "i revisits, t peek truth (Shift+T toggle), g ghost",
                "s trails, o paths, e clue heat, j json",
                "k csv history, l csv episode summary",
            ]
            for hint in hints:
                lbl = self.font_micro.render(hint, True, (190, 190, 205))
                self.win.blit(lbl, (panel_x, panel_y))
                panel_y += lbl.get_height()
        def draw(self) -> None:
            snapshot = self.current_snapshot()
            size = snapshot["world"]["size"]
            visited = {
                (item["cell"][0], item["cell"][1]): item["count"]
                for item in snapshot["knowledge"]["visited"]
            }
            known_clues = [tuple(c) for c in snapshot["knowledge"]["known_clues"]]
            world_clues = [tuple(c) for c in snapshot["world"]["clues"]]
            object_cell = tuple(snapshot["world"]["object"])
            robots = snapshot["robots"]
            prob_map = robots[0]["last_prob_map"] if robots and robots[0]["last_prob_map"] else [0.0] * (size * size)
            max_prob = max(prob_map) if prob_map else 0.0
            truth_visible = self.toggles["show_truth"] or (not self.toggles["show_truth"] and self.toggles["peek_truth"])
            viewing_latest = self.viewing_latest()
            self.win.fill((30, 30, 35))
            heat_cell = pygame.Surface((self.cell_px - 1, self.cell_px - 1), pygame.SRCALPHA)
            for y in range(size):
                for x in range(size):
                    top_left = self.world_to_screen((x, y), size)
                    rect = pygame.Rect(top_left[0], top_left[1], self.cell_px - 1, self.cell_px - 1)
                    vcount = visited.get((x, y), 0)
                    if self.toggles["show_visited"] and vcount > 0:
                        shade = clamp(40 + 35 * vcount, 40, 200)
                        pygame.draw.rect(self.win, (shade, shade, 220), rect)
                    else:
                        pygame.draw.rect(self.win, (50, 50, 60), rect)
                    if self.toggles["show_heatmap"] and prob_map:
                        idx = y * size + x
                        prob = prob_map[idx] if idx < len(prob_map) else 0.0
                        intensity = prob / max_prob if max_prob > 0 else 0.0
                        if self.toggles["contour"]:
                            steps = 5
                            intensity = round(intensity * (steps - 1)) / (steps - 1) if steps > 1 else intensity
                        intensity = clamp(intensity, 0.0, 1.0)
                        alpha = int(210 * intensity)
                        if alpha > 0:
                            heat_cell.fill((255, 140, 0, alpha))
                            self.win.blit(heat_cell, top_left)
                    pygame.draw.rect(self.win, (60, 60, 70), rect, 1)
                    if self.toggles["show_prob_numbers"] and prob_map:
                        idx = y * size + x
                        prob = prob_map[idx] if idx < len(prob_map) else 0.0
                        label = self.font_micro.render(f"{prob:.2f}", True, (15, 15, 20) if prob > max_prob * 0.6 else (235, 235, 240))
                        label_rect = label.get_rect(center=self.world_to_screen_center((x, y), size))
                        self.win.blit(label, label_rect)

                    if self.toggles["show_revisits"] and vcount > 1:
                        center = self.world_to_screen_center((x, y), size)
                        revisit_label = self.font_small.render(f"{vcount}", True, (255, 255, 255))
                        revisit_bg = pygame.Surface((revisit_label.get_width() + 4, revisit_label.get_height() + 2), pygame.SRCALPHA)
                        revisit_bg.fill((200, 80, 80, 180))
                        pygame.draw.rect(revisit_bg, (255, 255, 255), revisit_bg.get_rect(), 1)
                        bg_rect = revisit_bg.get_rect(center=(center[0], center[1] - 8))
                        self.win.blit(revisit_bg, bg_rect)
                        label_rect = revisit_label.get_rect(center=(center[0], center[1] - 8))
                        self.win.blit(revisit_label, label_rect)
            if truth_visible:
                now_ms = pygame.time.get_ticks()
                pulse = 0.5 + 0.5 * math.sin(now_ms / 250.0)
                obj_center = self.world_to_screen_center(object_cell, size)
                glow_radius = int(self.cell_px * (0.8 + 0.25 * pulse))
                halo_alpha = int(150 + 90 * pulse)
                glow_surface = pygame.Surface((self.cell_px * 2, self.cell_px * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (255, 0, 140, halo_alpha), (self.cell_px, self.cell_px), glow_radius)
                self.win.blit(glow_surface, (obj_center[0] - self.cell_px, obj_center[1] - self.cell_px))
                obj_rect = pygame.Rect(*self.world_to_screen(object_cell, size), self.cell_px - 1, self.cell_px - 1)
                border_radius = max(6, self.cell_px // 5)
                pygame.draw.rect(self.win, (255, 0, 140), obj_rect, border_radius=border_radius)
                inner_margin = max(3, self.cell_px // 4)
                inner_rect = obj_rect.inflate(-2 * inner_margin, -2 * inner_margin)
                if inner_rect.width > 0 and inner_rect.height > 0:
                    pygame.draw.rect(self.win, (255, 255, 255), inner_rect, border_radius=max(4, border_radius - 2))
                pygame.draw.rect(self.win, (90, 0, 90), obj_rect, width=3, border_radius=border_radius)
                clue_glow_alpha = int(120 + 70 * pulse)
                clue_radius = int(clamp(self.cell_px * 0.42, 4, self.cell_px))
                inner_radius = max(2, clue_radius - max(2, self.cell_px // 12))
                outline_width = max(2, self.cell_px // 14)
                for clue in world_clues:
                    center_pos = self.world_to_screen_center(clue, size)
                    clue_glow_surface = pygame.Surface((self.cell_px * 2, self.cell_px * 2), pygame.SRCALPHA)
                    clue_glow_radius = int(self.cell_px * (0.6 + 0.2 * pulse))
                    pygame.draw.circle(clue_glow_surface, (0, 255, 200, clue_glow_alpha), (self.cell_px, self.cell_px), clue_glow_radius)
                    self.win.blit(clue_glow_surface, (center_pos[0] - self.cell_px, center_pos[1] - self.cell_px))
                    pygame.draw.circle(self.win, (0, 235, 205), center_pos, clue_radius)
                    pygame.draw.circle(self.win, (0, 60, 70), center_pos, clue_radius, width=outline_width)
                    if inner_radius > 0:
                        pygame.draw.circle(self.win, (255, 255, 255), center_pos, inner_radius)
            if self.toggles["show_trails"]:
                for idx, robot in enumerate(robots):
                    color = self.robot_palette[idx % len(self.robot_palette)]
                    if self.selected_robot and self.selected_robot != robot["rid"]:
                        color = tuple(int(c * 0.4) for c in color)
                    pts = robot["path_history"][-60:]
                    if len(pts) >= 2:
                        trail_points = [
                            self.world_to_screen_center((p[0], p[1]), size)
                            for p in pts
                        ]
                        for i in range(1, len(trail_points)):
                            pygame.draw.line(self.win, tuple(int(c * 0.6) for c in color), trail_points[i - 1], trail_points[i], 2)
            if self.toggles["show_paths"]:
                for idx, robot in enumerate(robots):
                    color = self.robot_palette[idx % len(self.robot_palette)]
                    if self.selected_robot and self.selected_robot != robot["rid"]:
                        color = tuple(int(c * 0.4) for c in color)
                    path = robot["last_path"]
                    if len(path) >= 2:
                        pts = [
                            self.world_to_screen_center((p[0], p[1]), size)
                            for p in path
                        ]
                        pygame.draw.lines(self.win, color, False, pts, 2)
                        next_cell = robot["last_next_cell"]
                        if next_cell:
                            nx, ny = self.world_to_screen_center((next_cell[0], next_cell[1]), size)
                            pygame.draw.circle(self.win, color, (nx, ny), 6, 1)
            for idx, robot in enumerate(robots):
                rid = robot["rid"]
                color = self.robot_palette[idx % len(self.robot_palette)]
                goal_color = (255, 255, 255)
                if self.selected_robot and self.selected_robot != rid:
                    color = tuple(int(c * 0.6) for c in color)
                goal = robot["current_goal"]
                if goal:
                    gx, gy = goal
                    goal_top_left = self.world_to_screen((gx, gy), size)
                    goal_rect = pygame.Rect(goal_top_left[0], goal_top_left[1], self.cell_px, self.cell_px)
                    overlay = pygame.Surface((self.cell_px, self.cell_px), pygame.SRCALPHA)
                    overlay.fill((*color, 90))
                    self.win.blit(overlay, goal_rect.topleft)
                    pygame.draw.rect(self.win, goal_color, goal_rect.inflate(-8, -8), 2, border_radius=2)
                    if (gx, gy) not in visited:
                        pygame.draw.rect(self.win, (255, 220, 60), goal_rect, 2, border_radius=2)
                pos = tuple(robot["pos"])
                cell_top_left = self.world_to_screen(pos, size)
                body_rect = pygame.Rect(cell_top_left[0] + 4, cell_top_left[1] + 4, self.cell_px - 8, self.cell_px - 8)
                pygame.draw.rect(self.win, color, body_rect, border_radius=4)
                label = self.font_small.render(rid, True, (20, 20, 30))
                self.win.blit(label, label.get_rect(center=body_rect.center))
            collision_cells = self.collision_highlights if viewing_latest else [
                {"cell": (c[0], c[1]), "ttl": 0.5} for c in snapshot["collisions"]
            ]
            for item in collision_cells:
                cell = tuple(item["cell"])
                top_left = self.world_to_screen(cell, size)
                rect = pygame.Rect(top_left[0], top_left[1], self.cell_px - 1, self.cell_px - 1)
                pygame.draw.rect(self.win, (255, 80, 80), rect, 3)
            if self.toggles["show_contrib"]:
                contribs = self.compute_contributions(snapshot)
                if contribs:
                    block = 12
                    offset_x = self.margin + self.grid_px + 20
                    offset_y = self.margin + self.grid_px - (block * size) - 10
                    for clue, arr in contribs.items():
                        label = self.font_small.render(f"clue {clue}", True, (235, 235, 245))
                        self.win.blit(label, (offset_x, offset_y - 16))
                        max_val = max(arr) if arr else 1.0
                        for y in range(size):
                            for x in range(size):
                                val = arr[y * size + x]
                                if val <= 0.0 or max_val <= 0.0:
                                    continue
                                intensity = clamp(val / max_val, 0.0, 1.0)
                                alpha = int(200 * intensity)
                                cell_rect = pygame.Rect(
                                    offset_x + x * block,
                                    offset_y + (size - 1 - y) * block,
                                    block - 1,
                                    block - 1,
                                )
                                overlay = pygame.Surface((block - 1, block - 1), pygame.SRCALPHA)
                                overlay.fill((120, 200, 255, alpha))
                                self.win.blit(overlay, cell_rect.topleft)
                                pygame.draw.rect(self.win, (40, 40, 60), cell_rect, 1)
                        offset_y -= block * size + 24
            if self.toggles["show_ghost"] and snapshot["ghost_path"]:
                ghost_points = [
                    self.world_to_screen_center((cell[0], cell[1]), size)
                    for cell in snapshot["ghost_path"]
                ]
                if len(ghost_points) >= 2:
                    pygame.draw.lines(self.win, (200, 200, 255), False, ghost_points, 2)
            self.draw_sidebar(snapshot)
            self.draw_tooltip(snapshot)
            pygame.display.flip()

        def draw_tooltip(self, snapshot: Dict[str, Any]) -> None:
            mx, my = pygame.mouse.get_pos()
            size = snapshot["world"]["size"]
            cell = self.screen_to_world(mx, my, size)
            if cell is None:
                return
            cell_x, cell_y = cell
            visited = {
                (item["cell"][0], item["cell"][1]): item["count"]
                for item in snapshot["knowledge"]["visited"]
            }
            robots = snapshot["robots"]
            prob_map = robots[0]["last_prob_map"] if robots and robots[0]["last_prob_map"] else [0.0] * (size * size)
            idx = cell_y * size + cell_x
            prob_val = prob_map[idx] if idx < len(prob_map) else 0.0
            truth_visible = self.toggles["show_truth"] or (not self.toggles["show_truth"] and self.toggles["peek_truth"])
            world_clues = [tuple(c) for c in snapshot["world"]["clues"]]
            clue_lines = []
            for clue in world_clues:
                dist = manhattan((cell_x, cell_y), clue) if truth_visible else None
                desc = f"d={dist}" if dist is not None else "d=?"
                clue_lines.append(f"clue {clue}: {desc}")
            lines = [
                f"cell=({cell_x},{cell_y}) prob={prob_val:.3f}",
                f"visited={visited.get((cell_x, cell_y), 0)}",
            ] + clue_lines
            robot_here = [r["rid"] for r in robots if tuple(r["pos"]) == (cell_x, cell_y)]
            if robot_here:
                lines.append(f"robots={','.join(robot_here)}")
            tooltip_surface = pygame.Surface((240, 20 + 16 * len(lines)), pygame.SRCALPHA)
            tooltip_surface.fill((20, 20, 30, 210))
            pygame.draw.rect(tooltip_surface, (80, 80, 100), tooltip_surface.get_rect(), 1)
            ty = 6
            for line in lines:
                label = self.font_small.render(line, True, (235, 235, 245))
                tooltip_surface.blit(label, (8, ty))
                ty += 16
            px = mx + 12
            py = my + 12
            if px + tooltip_surface.get_width() > self.win.get_width():
                px = mx - tooltip_surface.get_width() - 12
            if py + tooltip_surface.get_height() > self.win.get_height():
                py = my - tooltip_surface.get_height() - 12
            self.win.blit(tooltip_surface, (px, py))

        def export_history_json(self) -> None:
            if not self.history:
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"viewer_snapshots_{timestamp}.json"
            path = os.path.join(os.getcwd(), filename)
            payload = {
                "config": {
                    "grid_size": self.cfg.grid_size,
                    "robots": self.cfg.robots,
                    "clues": self.cfg.clue_count,
                },
                "history": self.history,
                "event_log": self.event_log,
            }
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2)
            self.log_event("export", f"Saved {filename}")

        def export_history_csv(self) -> None:
            if not self.history:
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"viewer_snapshots_{timestamp}.csv"
            path = os.path.join(os.getcwd(), filename)
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["tick", "robot", "pos_x", "pos_y", "goal_x", "goal_y", "steps", "collision_replans", "goal_replans", "last_event", "last_reason"])
                for frame in self.history:
                    tick = frame["tick"]
                    for robot in frame["robots"]:
                        goal = robot["current_goal"] or [None, None]
                        writer.writerow([
                            tick,
                            robot["rid"],
                            robot["pos"][0],
                            robot["pos"][1],
                            goal[0],
                            goal[1],
                            robot["steps_taken"],
                            robot["collision_replans"],
                            robot["goal_replans"],
                            robot["last_event"],
                            robot["last_step_reason"],
                        ])
            self.log_event("export", f"Saved {filename}")

        def export_episode_summary_csv(self) -> None:
            if not self.robots:
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"viewer_episode_summary_{timestamp}.csv"
            path = os.path.join(os.getcwd(), filename)

            # Calculate episode metrics (same format as batch mode)
            self.update_team_knowledge()
            total_steps = sum(rb.steps_taken for rb in self.robots)
            metrics = collect_episode_result(self.world, self.know, self.robots, total_steps, self.found)
            collision_total = sum(metrics.collision_replans.values())
            goal_total = sum(metrics.goal_replans.values())
            replans_total = collision_total + goal_total

            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                # Same headers as batch mode
                cols = [
                    "episode", "grid", "robots", "clues", "steps_total",
                    "object_x", "object_y", "clue_locations", "found",
                    "discovered_clue_locations", "steps_before_first_clue", "steps_after_first_clue",
                    "robot_positions_at_first_clue", "collision_replans_total", "goal_replans_total",
                    "revisits", "system_revisits", "unique_cells"
                ]
                writer.writerow(cols)

                # Format data
                clue_locs_str = ";".join(f"({c[0]},{c[1]})" for c in metrics.clue_cells)
                discovered_locs_str = ";".join(f"({c[0]},{c[1]})" for c in metrics.discovered_clue_locations)
                robot_pos_str = ";".join(f"{rid}:({pos[0]},{pos[1]})" for rid, pos in sorted(metrics.robot_positions_at_first_clue.items()))

                # Single row with current episode data
                row = [
                    self.active_trial_index if self.active_trial_index is not None else 0,
                    self.cfg.grid_size, self.cfg.robots, self.cfg.clue_count, metrics.steps_total,
                    metrics.object_cell[0], metrics.object_cell[1], clue_locs_str, int(metrics.found),
                    discovered_locs_str, metrics.steps_before_first_clue, metrics.steps_after_first_clue,
                    robot_pos_str, metrics.collision_replans_total, metrics.goal_replans_total,
                    metrics.revisits, metrics.system_revisits, metrics.unique_cells
                ]
                writer.writerow(row)
            self.log_event("export", f"Saved {filename}")

        def save_all_trials_to_csv(self) -> None:
            """Save all completed trial results to CSV file."""
            if not self.completed_trials:
                print("No trials completed, skipping CSV export.")
                return

            csv_path = self.cfg.csv_path
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                cols = [
                    "episode","found","steps_total","steps_before_first_clue","steps_after_first_clue",
                    "robots","grid","object_x","object_y","clues","discovered_clues",
                    "revisits","system_revisits","collision_replans_total","goal_replans_total","replans_total","unique_cells"
                ]
                w.writerow(cols)

                for trial in self.completed_trials:
                    w.writerow(trial['row_values'])

            print(f"\nAll trial results saved to: {csv_path}")

        def display_episode_results(self) -> None:
            """Display final episode results in batch CSV format with labels."""
            if not self.robots:
                return

            self.update_team_knowledge()
            total_steps = sum(rb.steps_taken for rb in self.robots)
            metrics = collect_episode_result(self.world, self.know, self.robots, total_steps, self.found)
            collision_total = sum(metrics.collision_replans.values())
            goal_total = sum(metrics.goal_replans.values())
            replans_total = collision_total + goal_total

            # Save trial results for CSV export
            row_values = [
                self.active_trial_index if self.active_trial_index is not None else 0,
                int(metrics.found),
                metrics.steps_total,
                metrics.steps_before_first_clue,
                metrics.steps_after_first_clue,
                self.cfg.robots,
                self.cfg.grid_size,
                metrics.object_cell[0],
                metrics.object_cell[1],
                self.cfg.clue_count,
                metrics.discovered_clues,
                metrics.revisits,
                metrics.system_revisits,
                collision_total,
                goal_total,
                replans_total,
                metrics.unique_cells,
            ]

            self.completed_trials.append({
                'episode': self.active_trial_index if self.active_trial_index is not None else 0,
                'row_values': row_values
            })

            print("\n" + "="*60)
            print("EPISODE RESULTS (Batch CSV Format)")
            print("="*60)
            print(f"episode: {self.active_trial_index if self.active_trial_index is not None else 0}")
            print(f"found: {int(metrics.found)}")
            print(f"steps_total: {metrics.steps_total}")
            print(f"steps_before_first_clue: {metrics.steps_before_first_clue}")
            print(f"steps_after_first_clue: {metrics.steps_after_first_clue}")
            print(f"robots: {self.cfg.robots}")
            print(f"grid: {self.cfg.grid_size}")
            print(f"object_x: {metrics.object_cell[0]}")
            print(f"object_y: {metrics.object_cell[1]}")
            print(f"clues: {self.cfg.clue_count}")
            print(f"discovered_clues: {metrics.discovered_clues}")
            print(f"revisits: {metrics.revisits}")
            print(f"system_revisits: {metrics.system_revisits}")
            print(f"collision_replans_total: {collision_total}")
            print(f"goal_replans_total: {goal_total}")
            print(f"replans_total: {replans_total}")

            print("-" * 60)
            print("CSV Row:")
            csv_row = ",".join(str(v) for v in row_values)
            print(csv_row)
            print("="*60 + "\n")

        def handle_mouse(self, event: Any) -> None:
            if event.button != 1:
                return
            mx, my = event.pos
            snapshot = self.current_snapshot()
            size = snapshot["world"]["size"]
            cell = self.screen_to_world(mx, my, size)
            if cell is None:
                return
            cell_x, cell_y = cell
            for robot in snapshot["robots"]:
                if tuple(robot["pos"]) == (cell_x, cell_y):
                    if self.selected_robot == robot["rid"]:
                        self.selected_robot = None
                    else:
                        self.selected_robot = robot["rid"]
                    return
            self.selected_robot = None


        def handle_key(self, event: Any) -> None:
            key = event.key
            mods = event.mod
            if key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            if key == pygame.K_p:
                self.paused = not self.paused
                return
            if key == pygame.K_SPACE:
                if self.paused and not self.found:
                    self.advance_tick()
                else:
                    self.paused = not self.paused
                return
            if key == pygame.K_PERIOD:
                if self.history and self.history_view_index < len(self.history) - 1:
                    self.adjust_history(1)
                elif self.paused and not self.found:
                    self.advance_tick()
                return
            if key == pygame.K_COMMA:
                self.adjust_history(-1)
                return
            if key == pygame.K_LEFTBRACKET:
                self.viewer_fps = max(self.min_fps, self.viewer_fps - 0.5)
                self.cfg.viewer_fps = max(1, int(round(self.viewer_fps)))
                return
            if key == pygame.K_RIGHTBRACKET:
                self.viewer_fps = min(self.max_fps, self.viewer_fps + 0.5)
                self.cfg.viewer_fps = max(1, int(round(self.viewer_fps)))
                return
            if key == pygame.K_n:
                self.new_episode()
                return
            if key == pygame.K_r:
                self.new_episode(replay=True)
                return
            if key == pygame.K_h:
                self.toggles["show_heatmap"] = not self.toggles["show_heatmap"]
                return
            if key == pygame.K_v:
                self.toggles["show_visited"] = not self.toggles["show_visited"]
                return
            if key == pygame.K_b:
                self.toggles["show_prob_numbers"] = not self.toggles["show_prob_numbers"]
                return
            if key == pygame.K_c:
                self.toggles["contour"] = not self.toggles["contour"]
                return
            if key == pygame.K_t:
                if mods & pygame.KMOD_SHIFT:
                    self.toggles["show_truth"] = not self.toggles["show_truth"]
                    self.cfg.show_truth_in_viewer = self.toggles["show_truth"]
                    if self.toggles["show_truth"]:
                        self.toggles["peek_truth"] = False
                else:
                    self.toggles["peek_truth"] = not self.toggles["peek_truth"]
                return
            if key == pygame.K_g:
                self.toggles["show_ghost"] = not self.toggles["show_ghost"]
                return
            if key == pygame.K_s:
                self.toggles["show_trails"] = not self.toggles["show_trails"]
                return
            if key == pygame.K_o:
                self.toggles["show_paths"] = not self.toggles["show_paths"]
                return
            if key == pygame.K_e:
                self.toggles["show_contrib"] = not self.toggles["show_contrib"]
                return
            if key == pygame.K_i:
                self.toggles["show_revisits"] = not self.toggles["show_revisits"]
                return
            if key == pygame.K_j:
                self.export_history_json()
                return
            if key == pygame.K_k:
                self.export_history_csv()
                return
            if key == pygame.K_l:
                self.export_episode_summary_csv()
                return
            if key == pygame.K_HOME:
                if self.history:
                    self.history_view_index = 0
                    self.follow_history = False
                return
            if key == pygame.K_END:
                self.jump_to_latest()
                return

        def run(self) -> None:
            running = True
            while running:
                dt_ms = self.clock.tick(60)
                dt = dt_ms / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        self.handle_key(event)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_mouse(event)
                if self.metrics_plotter:
                    self.metrics_plotter.pump_events()
                if not running:
                    break
                if not self.paused and not self.found:
                    step_interval = 1.0 / max(self.viewer_fps, self.min_fps)
                    self.viewer_accumulator += dt
                    while self.viewer_accumulator >= step_interval:
                        self.advance_tick()
                        self.viewer_accumulator -= step_interval
                        if self.found:
                            break
                self.update_highlights(dt)
                self.draw()

            # Save all completed trials to CSV before exiting
            self.save_all_trials_to_csv()

    app = ViewerApp(cfg, trials)

    # Print which trial is running
    print(f"\n{'='*60}")
    print(f"Starting with Trial {cfg.start_trial}")
    print(f"{'='*60}\n")

    app.run()

    # Auto-increment trial counter for next run
    counter_file = "viewer_trial_counter_B_async.txt"
    next_trial = (cfg.start_trial + 1) % len(trials)
    with open(counter_file, "w") as f:
        f.write(str(next_trial))
    print(f"\nNext run will start with Trial {next_trial}")
    print(f"(To reset, delete {counter_file} or edit start_trial in Config)")
# ------------------------
# CLI
# ------------------------
def parse_args(argv=None) -> Config:
    base_cfg = Config()
    p = argparse.ArgumentParser(description="Pololu Search Simulator v3 async (B-00 forward-priority)")
    p.add_argument("--mode", choices=["batch", "view"], default=base_cfg.mode)
    p.add_argument("--grid", type=int, default=base_cfg.grid_size)
    p.add_argument("--robots", type=int, default=base_cfg.robots, choices=[1, 2, 4])
    p.add_argument("--clues", type=int, default=base_cfg.clue_count)
    p.add_argument("--episodes", type=int, default=base_cfg.episodes)
    p.add_argument("--csv", dest="csv_path", default=base_cfg.csv_path)
    p.add_argument("--reward-factor", type=float, default=base_cfg.reward_factor)
    p.add_argument("--center-step", type=float, default=base_cfg.center_step)
    p.add_argument("--visited-step-penalty", type=float, default=base_cfg.visited_step_penalty)
    p.add_argument("--intent-penalty", type=float, default=base_cfg.intent_penalty)
    p.add_argument("--viewer-fps", type=int, default=base_cfg.viewer_fps)
    p.add_argument("--cell-size", type=int, default=base_cfg.cell_px,
                   help="Pixels per grid cell in viewer (default: 40)")
    p.set_defaults(show_truth=None)
    truth_group = p.add_mutually_exclusive_group()
    truth_group.add_argument("--show-truth", dest="show_truth", action="store_true",
                             help="Always draw object/clue truth overlay (default).")
    truth_group.add_argument("--hide-truth", dest="show_truth", action="store_false",
                             help="Hide object/clue truth overlay.")
    p.add_argument("--trials-in", dest="trials_in", default=base_cfg.trials_in, help="Path to JSON list of precomputed trials")
    p.add_argument("--trials-out", dest="trials_out", default=base_cfg.trials_out, help="Write generated trials to this JSON path")
    p.add_argument("--scenarios-csv", dest="scenarios_csv", default=base_cfg.scenarios_csv, help="Path to CSV file with object/clue scenarios from generator")
    p.add_argument("--async-step-mean", type=float, default=base_cfg.async_step_mean,
                   help="Mean wall-clock seconds between robot updates in the async scheduler (default: 1.60).")
    p.add_argument("--async-step-jitter", type=float, default=base_cfg.async_step_jitter,
                   help="Uniform jitter applied to each robot update interval (default: +/-0.10).")
    p.add_argument("--async-min-delay", type=float, default=base_cfg.async_min_delay,
                   help="Minimum time between updates for any robot (default: 1.50).")
    p.add_argument("--async-max-delay", type=float, default=base_cfg.async_max_delay,
                   help="Maximum time between updates for any robot (default: 1.70; set <=0 to disable clamp).")
    p.add_argument("--async-initial-spread", type=float, default=base_cfg.async_initial_spread,
                   help="Multiplier controlling how far apart initial robot wake-ups are spread.")
    p.add_argument("--async-tick-span", type=float, default=None,
                   help="World time advanced per viewer tick (default: 0.25 for async visualization).")
    p.add_argument("--async-turn-quarter", type=float, default=base_cfg.async_turn_quarter,
                   help="Seconds required for a single 90-degree turn (default: 0.3).")
    p.add_argument("--intent-sync-delay", type=float, default=base_cfg.intent_sync_delay,
                   help="Seconds to wait after publishing intent before moving (default: 0.05).")
    p.add_argument("--async-seed", type=int, default=base_cfg.async_seed,
                   help="Random seed for async timing jitter (omit for nondeterministic runs).")
    p.add_argument("--comm-delay-ms", type=float, default=base_cfg.comm_delay_s * 1000.0,
                   help="Communication delay applied to inter-robot messages in milliseconds (default: 150).")
    a = p.parse_args(argv)
    show_truth = base_cfg.show_truth_in_viewer if a.show_truth is None else a.show_truth

    async_min_delay = max(0.0, a.async_min_delay)
    async_step_mean = max(async_min_delay, a.async_step_mean)
    async_step_jitter = max(0.0, a.async_step_jitter)
    async_max_delay = a.async_max_delay
    if async_max_delay > 0:
        async_max_delay = max(async_min_delay, async_max_delay)
    else:
        async_max_delay = async_step_mean
    async_initial_spread = max(0.0, a.async_initial_spread)
    async_turn_quarter = a.async_turn_quarter if (a.async_turn_quarter is not None and a.async_turn_quarter > 0) else max(async_step_mean, async_min_delay)
    intent_sync_delay = max(0.0, a.intent_sync_delay)

    if a.async_tick_span is not None and a.async_tick_span > 0:
        async_tick_span = a.async_tick_span
    else:
        async_tick_span = base_cfg.async_tick_span
    if async_tick_span <= 0:
        async_tick_span = max(0.05, base_cfg.async_tick_span)
    comm_delay_s = max(0.0, a.comm_delay_ms / 1000.0)

    return Config(
        grid_size=a.grid,
        robots=a.robots,
        clue_count=a.clues,
        reward_factor=a.reward_factor,
        center_step=a.center_step,
        visited_step_penalty=a.visited_step_penalty,
        intent_penalty=a.intent_penalty,
        episodes=a.episodes,
        mode=a.mode,
        csv_path=a.csv_path,
        viewer_fps=a.viewer_fps,
        cell_px=a.cell_size,
        show_truth_in_viewer=show_truth,
        trials_in=a.trials_in,
        trials_out=a.trials_out,
        scenarios_csv=a.scenarios_csv,
        async_step_mean=async_step_mean,
        async_step_jitter=async_step_jitter,
        async_min_delay=async_min_delay,
        async_max_delay=async_max_delay,
        async_initial_spread=async_initial_spread,
        async_tick_span=async_tick_span,
        async_turn_quarter=async_turn_quarter,
        async_seed=a.async_seed,
        intent_sync_delay=intent_sync_delay,
        comm_delay_s=comm_delay_s,
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



























