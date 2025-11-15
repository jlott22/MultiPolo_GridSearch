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
# ===========================================================
# Pololu 3pi+ 2040 OLED — Coordinated Search (UART → ESP32 → MQTT)
# ===========================================================
# Runs on the Pololu 3pi+ 2040 OLED using MicroPython.
# Communication uses simple text frames over UART; an attached ESP32 relays
# those frames to MQTT topics.
#
# Behavior overview:
#   * Before any clue is found the robot sweeps its half of the grid in a
#     lawn‑mower pattern, nudged outward by a small centre‑ward cost.
#   * After a clue appears, A* planning pursues cells with the highest
#     probability scores.
#   * The next intended cell is published so peers can yield and avoid
#     collisions.
#   * Bump sensors detect the object; on a bump both robots halt and report.
#   * A clue is any intersection where the centered line sensor reads white.
#
# Threads:
#   * A background movement thread handles forward motion while the main thread processes UART and coordinates movement.
#   * The main thread plans paths and moves the robot, always stopping the
#     motors if the program exits unexpectedly.
#
# Tuning hints:
#   * Set UART pins and baud rate to match the hardware.
#   * Calibrate line sensors and adjust cfg.MIDDLE_WHITE_THRESH accordingly.
#   * Tune yaw timings (cfg.YAW_90_MS / cfg.YAW_180_MS) for your platform.
# TODO: fix counting start cell in unique cells
# ===========================================================

import time
import _thread
import heapq
import sys
import gc
from array import array
from machine import UART, Pin
from pololu_3pi_2040_robot import robot
from pololu_3pi_2040_robot.extras import editions
from pololu_3pi_2040_robot.buzzer import Buzzer

# -----------------------------
# Robot identity & start pose
# -----------------------------
ROBOT_ID = "00"  # set to "00", "01", "02", or "03" at deployment
GRID_SIZE = 10
GRID_CENTER = (GRID_SIZE - 1) / 2

DEBUG_LOG_FILE = "debug-log-00.txt"

METRICS_LOG_FILE = "metrics-log-00A.txt"
BOOT_TIME_MS = time.ticks_ms()
METRIC_START_TIME_MS = None  # set after first post-calibration intersection
start_signal = False  # set when hub command received
intersection_visits = {}
unique_cells_count = 0       # cells first visited by this robot for the entire team
system_visits = {}              # all cells visited by any robot (for tracking system_revisits)
intersection_count = 0          # steps taken by this robot
system_revisits = 0             # this robot's revisits to cells visited by ANY robot
yield_count = 0                 # times this robot yielded an intended move
path_replan_count = 0           # times we replanned due to collision avoidance
goal_replan_count = 0           # times our goal changed and we replanned
FIRST_CLUE_TIME_MS = None       # ms from start to first clue (by any robot)
FIRST_CLUE_POSITION = None      # this robot's position when first clue found (by any robot)
object_location = None          # set when object is found
system_clues_found = 0          # total unique clues found by all robots
steps_after_first_clue = 0      # steps taken after first clue was found

_metrics_logged = False
_metrics_cache = None

buzzer = None  # replaced after hardware initialization

# Energy/Time metrics
motor_time_ms = 0              # cumulative ms motors were commanded non-zero
_motor_start_ms = None         # internal tracker for motor activity

def finalize_motor_time(now_ticks=None):
    """Ensure motor_time_ms captures any active span before sampling metrics."""
    global _motor_start_ms, motor_time_ms
    if _motor_start_ms is not None:
        if now_ticks is None:
            now_ticks = time.ticks_ms()
        motor_time_ms += time.ticks_diff(now_ticks, _motor_start_ms)
        _motor_start_ms = None


def busy_timer_reset():
    """Start a fresh busy-time measurement for the current control loop."""
    global _busy_start_us, _busy_accum_us
    _busy_accum_us = 0
    _busy_start_us = time.ticks_us()


def busy_timer_pause():
    """Accumulate elapsed busy time and pause the timer."""
    global _busy_start_us, _busy_accum_us
    if _busy_start_us is not None:
        now_us = time.ticks_us()
        _busy_accum_us += time.ticks_diff(now_us, _busy_start_us)
        _busy_start_us = None


def busy_timer_resume():
    """Resume the busy-time timer after a pause."""
    global _busy_start_us
    _busy_start_us = time.ticks_us()


def busy_timer_value_ms():
    """Return the current busy time in milliseconds, pausing measurement."""
    global _busy_start_us, _busy_accum_us
    if _busy_start_us is not None:
        now_us = time.ticks_us()
        _busy_accum_us += time.ticks_diff(now_us, _busy_start_us)
        _busy_start_us = None
    return _busy_accum_us // 1000


def update_mem_headroom():
    """Refresh current free heap measurement and track the lowest observed value."""
    global mem_free_min
    current = gc.mem_free()
    if current < mem_free_min:
        mem_free_min = current
    return current


# Simple energy tracking - message counters only
position_msgs_sent = 0
visited_msgs_sent = 0
clue_msgs_sent = 0
object_msgs_sent = 0
intent_msgs_sent = 0
position_msgs_received = 0
visited_msgs_received = 0
clue_msgs_received = 0
object_msgs_received = 0
intent_msgs_received = 0
# Time metrics
busy_ms = 0                 # cumulative compute time spent outside motion/sleeps (ms)
mem_free_min = gc.mem_free()  # lowest observed free heap bytes

_busy_start_us = None       # internal timer start (microseconds)
_busy_accum_us = 0          # accumulated busy time (microseconds)


def log_error(message):
    """Log errors with a timestamp and play a low buzzer tone."""
    elapsed_ms = time.ticks_diff(time.ticks_ms(), BOOT_TIME_MS)
    try:
        with open(DEBUG_LOG_FILE, "a") as _fp:
            _fp.write(f"{elapsed_ms} ERROR: {message}\n")
    except (OSError, MemoryError):
        pass
    try:
        if buzzer is not None:
            buzzer.play("O2c16")
    except Exception:
        pass


def safe_assert(condition, message):
    if not condition:
        log_error(message)
        raise AssertionError(message)


def record_intersection(x, y):
    """Track intersection visits and system revisit counts."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "intersection out of range")
    global intersection_count, system_revisits, system_visits, unique_cells_count
    intersection_count += 1
    if first_clue_seen:
        global steps_after_first_clue
        steps_after_first_clue += 1
    key = (x, y)

    # Track system-wide revisits (ANY robot visited before)
    first_visit = key not in system_visits
    if not first_visit:
        system_revisits += 1

    # Update system-wide visits (this robot's contribution)
    system_visits[key] = system_visits.get(key, 0) + 1
    if first_visit:
        unique_cells_count += 1

    # Update individual visit tracking
    if key in intersection_visits:
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1


def simple_energy_metrics(elapsed_ms):
    """Calculate time metrics only - power can be computed offline if needed."""
    # Compute time = everything except motor time
    compute_time_ms = max(0, elapsed_ms - motor_time_ms)

    # Message totals
    total_msgs_sent = (position_msgs_sent + visited_msgs_sent + clue_msgs_sent +
                      object_msgs_sent + intent_msgs_sent)
    total_msgs_received = (position_msgs_received + visited_msgs_received +
                          clue_msgs_received + object_msgs_received + intent_msgs_received)

    return {
        'motor_time_ms': motor_time_ms,
        'compute_time_ms': compute_time_ms,
        'msgs_sent': total_msgs_sent,
        'msgs_received': total_msgs_received,
    }

def manhatt_dist_metric(clue_location,bot_location):
    """Calculate Manhattan distance between first clue and bot location when first clue discovered."""
    return abs(clue_location[0] - bot_location[0]) + abs(clue_location[1] - bot_location[1])

def metrics_log():
    """Write summary metrics for the search run and return them."""
    global unique_cells_count, busy_ms, mem_free_min, _metrics_logged, _metrics_cache
    if _metrics_logged and _metrics_cache is not None:
        return _metrics_cache
    start = METRIC_START_TIME_MS if METRIC_START_TIME_MS is not None else BOOT_TIME_MS
    now = time.ticks_ms()
    finalize_motor_time(now)
    elapsed_ms = time.ticks_diff(now, start)
    unique_cells = unique_cells_count
    compute_time_ms = max(0, elapsed_ms - motor_time_ms)
    dist_from_first_clue = manhatt_dist_metric(clues[0],FIRST_CLUE_POSITION) if FIRST_CLUE_POSITION is not None and len(clues)>0 else -1

    # Calculate time metrics
    energy = simple_energy_metrics(elapsed_ms)

    metrics = {
        "robot_id": ROBOT_ID,
        "object_location": object_location,
        "clue_locations": clues,
        "elapsed_ms": elapsed_ms,
        "motor_time_ms": motor_time_ms,
        "compute_time_ms": compute_time_ms,
        "busy_ms": busy_ms,
        "mem_free_min": mem_free_min,
        "steps": intersection_count,
        "first_clue_time_ms": FIRST_CLUE_TIME_MS if FIRST_CLUE_TIME_MS is not None else -1,
        "dist_from_1st_clue": dist_from_first_clue,
        "steps_after_first_clue": steps_after_first_clue,
        "system_clues_found": system_clues_found,
        "system_revisits": system_revisits,
        "unique_cells": unique_cells,
        "yields": yield_count,
        "goal_replans": goal_replan_count,
        "msgs_sent": energy['msgs_sent'],
        "msgs_received": energy['msgs_received'],
        "path_replans": path_replan_count,
    }

    fieldnames = [
        "robot_id",
        "object_location",
        "clue_locations",
        "elapsed_ms",
        "motor_time_ms",
        "compute_time_ms",
        "busy_ms",
        "mem_free_min",
        "steps",
        "first_clue_time_ms",
        "dist_from_1st_clue",
        "steps_after_first_clue",
        "system_clues_found",
        "system_revisits",
        "unique_cells",
        "yields",
        "goal_replans",
        "msgs_sent",
        "msgs_received",
        "path_replans",
    ]

    try:
        try:
            with open(METRICS_LOG_FILE) as _fp:
                write_header = _fp.read(1) == ""
        except OSError:
            write_header = True
        with open(METRICS_LOG_FILE, "a") as _fp:
            if write_header:
                _fp.write(",".join(fieldnames) + "\n")
            _fp.write(",".join(str(metrics[f]) for f in fieldnames) + "\n")
    except OSError:
        pass
    _metrics_cache = metrics
    _metrics_logged = True
    return metrics


try:
    open(DEBUG_LOG_FILE, "a").close()
except OSError:
    pass

try:
    open(METRICS_LOG_FILE, "a").close()
except OSError:
    pass

# Starting position & heading (grid coordinates, cardinal heading)
# pos = (x, y)    heading = (dx, dy) where (0,1)=N, (1,0)=E, (0,-1)=S, (-1,0)=W
START_CONFIG = {
    "00": ((0, 0), (0, 1)),                       # SW corner, facing north
    "01": ((GRID_SIZE - 1, GRID_SIZE - 1), (0, -1)),  # NE corner, facing south
    "02": ((0, GRID_SIZE - 1), (1, 0)),           # NW corner, facing east
    "03": ((GRID_SIZE - 1, 0), (-1, 0)),          # SE corner, facing west
}
DIRS4 = ((0, 1), (1, 0), (0, -1), (-1, 0))

try:
    START_POS, START_HEADING = START_CONFIG[ROBOT_ID]
except KeyError as e:
    raise ValueError("ROBOT_ID must be one of '00', '01', '02', or '03'") from e
safe_assert(0 <= START_POS[0] < GRID_SIZE and 0 <= START_POS[1] < GRID_SIZE,
            "start position out of bounds")

# UART0 for ESP32 communication (TX=GP28, RX=GP29)
uart = UART(0, baudrate=115200, tx=28, rx=29)

# -----------------------------
# Grid / Maps / Shared State
# -----------------------------
# Grid cell states
CELL_UNSEARCHED = 0
CELL_OBSTACLE   = 1  # object or peer reservation
CELL_SEARCHED   = 2

grid = bytearray(GRID_SIZE * GRID_SIZE)
prob_map = array('f', [1 / (GRID_SIZE * GRID_SIZE)] * (GRID_SIZE * GRID_SIZE))
REWARD_FACTOR = 5
clues = []                            # list of (x, y) clue cells

# Preallocated arrays for A* planning
# ----------------------------------
# Parent indices and path costs for each cell are stored here. Reusing these
# arrays each planning cycle avoids repeated allocations, which are expensive
# on MicroPython.
came_from = array('i', [-1] * (GRID_SIZE * GRID_SIZE))
cost_so_far = array('f', [0.0] * (GRID_SIZE * GRID_SIZE))
frontier = []


def idx(x, y):
    """Convert Cartesian (x, y) to linear index in map arrays."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "idx out of range")
    return (GRID_SIZE - 1 - y) * GRID_SIZE + x


pos = [START_POS[0], START_POS[1]]    # current grid position
heading = (START_HEADING[0], START_HEADING[1])

# Flags used by threads for clean exits
running = True                         # global run flag
found_object = False                   # set True on bump or peer alert
first_clue_seen = False                # once True, disable lawn‑mower bias
move_forward_flag = False

# Intent data shared by peers
peer_intent = {}  # peer_id -> (x, y) reservation
peer_pos = {}     # peer_id -> (x, y) last reported position
peer_goal = {}    # peer_id -> (x, y) goal reservation
current_goal = None  # our reserved goal cell
last_visited_from_sender = {}  # sender_id -> (x, y) to detect duplicate visited messages

# -----------------------------
# Cost shaping for early sweeping pattern
# A small cost per step toward the center keeps robots sweeping their region
# before clues are discovered. It must exceed the TURN_COST (1.0).
TURN_COST = 1.0
CENTER_STEP = 0.4

# -----------------------------
# Motion configuration
# -----------------------------
class MotionConfig:
    def __init__(self):
        self.MIDDLE_WHITE_THRESH = 200  # center sensor threshold for "white" (tune by calibration)
        self.VISITED_STEP_PENALTY = 4
        self.KP = 0.5                # proportional gain around LINE_CENTER
        self.CALIBRATE_SPEED = 1130  # speed to rotate when calibrating
        self.BASE_SPEED = 800        # nominal wheel speed
        self.MIN_SPD = 400           # clamp low (avoid stall)
        self.MAX_SPD = 1200          # clamp high
        self.LINE_CENTER = 2000      # weighted position target (0..4000)
        self.BLACK_THRESH = 600      # calibrated "black" threshold (0..1000)
        self.STRAIGHT_CREEP = 900    # forward speed while "locked" straight
        self.START_LOCK_MS = 300     # hold straight this long after function starts
        self.TURN_SPEED = 1000
        self.YAW_90_MS = 0.3
        self.YAW_180_MS = 0.6

cfg = MotionConfig()

# Intent settings
INTENT_PENALTY = 8.0     # strong penalty to avoid stepping into the other's reserved or occupied cell

#UART handling globals
# ---------- ring buffer ----------
RB_SIZE = 1024
buf = bytearray(RB_SIZE)
head = 0
tail = 0
DELIM = ord('-')

# ---------- message builder ----------
MSG_BUF_SIZE = 256
msg_buf = bytearray(MSG_BUF_SIZE)
msg_len = 0

# ---------- outbound buffer ----------
TX_BUF_SIZE = 64
tx_buf = bytearray(TX_BUF_SIZE)

def _write_int(buf, idx, val):
    """Write an integer as ASCII into buf starting at idx.

    Returns the new index after writing."""
    if val < 0:
        buf[idx] = ord('-')
        idx += 1
        val = -val
    if val == 0:
        buf[idx] = ord('0')
        return idx + 1
    # Determine number of digits
    tmp = val
    digits = 0
    while tmp:
        tmp //= 10
        digits += 1
    end = idx + digits
    for _ in range(digits):
        buf[end - 1] = ord('0') + (val % 10)
        val //= 10
        end -= 1
    return idx + digits

# -----------------------------
# Hardware interfaces
# -----------------------------
motors = robot.Motors()
line_sensors = robot.LineSensors()
bump = robot.BumpSensors()
rgb_leds = robot.RGBLEDs()
rgb_leds.set_brightness(10)
buzzer = Buzzer()

# ===========================================================
# Utility: Motors & Stop Control
# ===========================================================

RED   = (230, 0, 0)
GREEN = (0, 230, 0)
BLUE = (0, 0, 230)
OFF   = (0, 0, 0)

def flash_LEDS(color, n):
    for _ in range(n):
        for led in range(6):
            rgb_leds.set(led, color)  # reuses same tuple, no new allocation
        rgb_leds.show()
        time.sleep_ms(100)
        for led in range(6):
            rgb_leds.set(led, OFF)
        rgb_leds.show()
        time.sleep_ms(100)
        
def buzz(event):
    """
    Play short chirps for turn, intersection, clue,
    and a longer sequence for object.
    """
    if event == "turn":
        buzzer.play("O5c16")            # short high chirp
    elif event == "intersection":
        buzzer.play("O4g16")            # short mid chirp
    elif event == "clue":
        buzzer.play("O6e16")            # short very high chirp
    elif event == "object":
        buzzer.play("O4c8e8g8c5")       # longer sequence, rising melody


        
flash_LEDS(GREEN,1)

def set_speeds(left, right):
    """Wrapper to track motor active time before delegating to hardware."""
    global _motor_start_ms
    if left != 0 or right != 0:
        if _motor_start_ms is None:
            _motor_start_ms = time.ticks_ms()
    else:
        finalize_motor_time()
    motors.set_speeds(left, right)


def motors_off():
    """Hard stop both wheels (safety: call in finally/stop paths)."""
    set_speeds(0, 0)

def stop_all():
    """
    Idempotent global stop:
      - Set flags so all loops/threads exit
      - Ensure motors are off
      - Set a green LED to indicate finished
    """
    global running
    running = False
    motors_off()
    summary = metrics_log()
    publish_result(summary)

def stop_and_alert_object():
    """
    Called when THIS robot detects the object via bump.
    Publishes alert and performs a global stop.

    The robot may bump into the object before reaching the next
    intersection, leaving ``pos`` pointing to the last intersection it
    successfully crossed.  Report the object at the *next* intersection in
    the current heading direction so external consumers know where it is.
    """
    global object_location, found_object, intersection_count, steps_after_first_clue
    global intersection_visits, system_visits, system_revisits, unique_cells_count
    next_x = pos[0] + heading[0]
    next_y = pos[1] + heading[1]
    object_location = (next_x, next_y)
    key = (next_x, next_y)
    first_visit = key not in system_visits
    if not first_visit:
        system_revisits += 1
        system_visits[key] += 1
    else:
        system_visits[key] = 1
        unique_cells_count += 1
    if key in intersection_visits:
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1
    publish_object(next_x, next_y)
    buzz('object')
    found_object = True
    intersection_count += 1
    steps_after_first_clue += 1
    stop_all()
    flash_LEDS(BLUE, 1)

flash_LEDS(GREEN,1)
# ===========================================================
# UART Messaging
# Format: "<topic#>:<payload>\n"
# position = 1, visited = 2, clue = 3, alert = 4, intent = 5, result = 6, goal = 7
# Examples:
#   001.3,4-  robot 00 position (x,y only)
#   003.5,2-  robot 00 clue at (5,2)
#   005.7,8-  robot 00 intent/reservation at (7,8)
#   007.9,1-  robot 00 goal reservation at (9,1)
# ===========================================================
def uart_send(topic, payload_len):
    """Send the prepared message in tx_buf with topic and payload_len."""
    tx_buf[0] = ord(topic)
    tx_buf[1] = ord('.')
    tx_buf[payload_len + 2] = ord('-')
    uart.write(tx_buf[:payload_len + 3])

def publish_position():
    """Publish current pose (for UI/diagnostics)."""
    global position_msgs_sent
    position_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, pos[0])
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, pos[1])
    uart_send('1', i - 2)

def publish_visited(x, y):
    """Publish that we visited cell (x,y)."""
    global visited_msgs_sent
    visited_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('2', i - 2)

def publish_clue(x, y):
    """Publish a clue at (x,y)."""
    global clue_msgs_sent
    clue_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('3', i - 2)

def publish_object(x, y):
    """Publish that we found the object at (x,y)."""
    global object_msgs_sent
    object_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('4', i - 2)

def publish_intent(x, y):
    """
    Publish our intended next cell (reservation).
    Other robots will avoid stepping into this cell until a new intent is published.
    """
    global intent_msgs_sent
    intent_msgs_sent += 1
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('5', i - 2)


def publish_goal(x, y):
    """Publish our reserved goal cell."""
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('7', i - 2)

def publish_result(msg):
    """Publish final search metrics or result to the hub."""
    # ``msg`` can be numeric; ensure it is converted to string before
    # concatenation to avoid ``TypeError: can't convert 'int' object to str``.
    uart.write("6." + str(msg) + "-")

def handle_msg(line):
    """
    Parse and apply incoming messages from the other robot or hub.

    Accepts:
    011.3,4-       # topic 1: position (x,y only)
    002.3,4-       # topic 2: visited
    003.5,2-       # topic 3: clue
    004.6,1-       # topic 4: object/alert
    005.7,2-       # topic 5: intent
    007.2,3-       # topic 7: goal reservation
    996.1-         # topic 6: hub command

    Ignores:
      - other status fields we don't currently need
    """
    global peer_intent, peer_pos, peer_goal, current_goal, first_clue_seen, object_location, start_signal, found_object, FIRST_CLUE_TIME_MS, goal_replan_count

    # Minimal parsing: "<sender>/<topic>:<payload>"
    try:
        left, payload = line.split(".", 1)
        if len(left) < 3:
            return
        sender = left[0:2]
        topic  = left[2]
    except ValueError:
        return

    if topic == "2":  #visited
        global visited_msgs_received, system_visits
        visited_msgs_received += 1
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            # Check for duplicate message from same sender (heartbeat spam filter)
            if last_visited_from_sender.get(sender) == (x, y):
                # Same cell from same sender - update grid but don't process further
                i = idx(x, y)
                grid[i] = CELL_SEARCHED
                prob_map[i] = 0.0
                return

            # New cell from this sender - update tracking
            last_visited_from_sender[sender] = (x, y)

            # Track system-wide visits for system_revisits metric
            key = (x, y)
            system_visits[key] = system_visits.get(key, 0) + 1

            # Process normally
            i = idx(x, y)
            grid[i] = CELL_SEARCHED
            prob_map[i] = 0.0
            if current_goal == (x, y) and not (pos[0] == x and pos[1] == y):
                current_goal = None

    elif topic == "3":   #clue
        global clue_msgs_received, system_clues_found, FIRST_CLUE_POSITION
        clue_msgs_received += 1
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            clue = (x, y)
            if clue not in clues:
                clues.append(clue)
                system_clues_found += 1
                first_clue_seen = True
                if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                    FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                    FIRST_CLUE_POSITION = (pos[0], pos[1])
                update_prob_map()
                gc.collect()

    elif topic == "4": #object
        # Peer found the object → stop immediately
        global object_msgs_received
        object_msgs_received += 1
        try:
            x, y = map(int, payload.split(","))
            object_location = (x, y)
        except ValueError:
            object_location = None
        found_object = True
        stop_all()

    elif topic == "1": #position
        global position_msgs_received
        position_msgs_received += 1
        try:
            ox, oy = map(int, payload.split(","))
        except ValueError:
            return
        if not (0 <= ox < GRID_SIZE and 0 <= oy < GRID_SIZE):
            return
        prev = peer_pos.get(sender)
        if prev and prev != (ox, oy):
            px, py = prev
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                if peer_intent.get(sender) != (px, py):
                    grid[idx(px, py)] = CELL_SEARCHED
        peer_pos[sender] = (ox, oy)
        grid[idx(ox, oy)] = CELL_OBSTACLE

    elif topic == "5": #intent
        global intent_msgs_received
        intent_msgs_received += 1
        try:
            ix, iy = map(int, payload.split(","))
        except ValueError:
            return
        if not (0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE):
            return
        prev = peer_intent.get(sender)
        if prev and prev != (ix, iy):
            px, py = prev
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                if peer_pos.get(sender) != (px, py):
                    grid[idx(px, py)] = CELL_SEARCHED
        peer_intent[sender] = (ix, iy)
        grid[idx(ix, iy)] = CELL_OBSTACLE
    elif topic == "7": #goal reservation
        try:
            gx, gy = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            peer_goal[sender] = (gx, gy)
            if current_goal == (gx, gy):
                # our goal is taken; force replanning
                current_goal = None
    elif topic == "6":  # hub command
        if payload.strip() == "1":
            start_signal = True

# ---------- ring buffer helpers ----------
def rb_put_byte(b):
    """Push one byte into the ring buffer."""
    global tail, head
    buf[tail] = b
    nxt = (tail + 1) % RB_SIZE
    if nxt == head:                # buffer full, drop oldest
        head = (head + 1) % RB_SIZE
    tail = nxt

def rb_pull_into_msg():
    """Pull bytes into message buffer until '-' is found."""
    global head, tail, msg_len
    if head == tail:
        return None
    while head != tail:
        b = buf[head]
        head = (head + 1) % RB_SIZE
        if b == DELIM:  # complete frame
            s = msg_buf[:msg_len].decode('utf-8', 'ignore').strip()
            msg_len = 0
            return s
        if msg_len < MSG_BUF_SIZE:
            msg_buf[msg_len] = b
            msg_len += 1
    return None

# ---------- UART service ----------
def uart_service():
    """Read and parse any complete messages from UART."""
    data = uart.read()     # returns None or bytes object
    if not data:
        return
    for b in data:         # iterate over bytes
        rb_put_byte(b)
    while True:
        msg = rb_pull_into_msg()
        if msg is None:
            break
        handle_msg(msg)

# ===========================================================
# Sensing & Motion
# ===========================================================
flash_LEDS(GREEN,1)
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def move_forward_one_cell():
    """
    Drive forward following the line until an intersection is detected:
      - T or + intersections: trigger if either outer sensor is black.
      - Require 3 consecutive qualifying reads (debounce).
      - On first candidate, lock steering straight (no P-correction)
        until intersection is confirmed → avoids grabbing side lines.
      - Also hold a 0.5 s straight "roll-through" at start to clear
        the cross you’re sitting on before re-engaging P-control.
    Returns:
      True  -> reached an intersection (no bump)
      False -> stopped due to bump or external stop condition
    """
    global move_forward_flag
    first_loop = False
    lock_release_time = time.ticks_ms() #flag to reset start lock time
    #outter infinite loop to keep thread check for activation
    while running:
        
        while move_forward_flag:
            # 1) Safety/object check
            if first_loop:
                # Initial lock to roll straight for half a second
                lock_release_time = time.ticks_add(time.ticks_ms(), cfg.START_LOCK_MS)
                first_loop = False

            # 3) During initial lock window, always drive straight
            if time.ticks_diff(time.ticks_ms(), lock_release_time) < 0:
                set_speeds(cfg.STRAIGHT_CREEP, cfg.STRAIGHT_CREEP)
                continue
            
            # 2) Read sensors
            readings = line_sensors.read_calibrated()
            
            if readings[0] >= cfg.BLACK_THRESH or readings[4] >= cfg.BLACK_THRESH:
                motors_off()
                flash_LEDS(GREEN,1)
                move_forward_flag = False
                first_loop = True
                break
            
            bump.read()
            if bump.left_is_pressed() or bump.right_is_pressed():
                stop_and_alert_object()
                motors_off()
                move_forward_flag = False
                break    

            # 6) Normal P-control when not locked
            total = readings[0] + readings[1] + readings[2] + readings[3] + readings[4]
            if total == 0:
                set_speeds(cfg.STRAIGHT_CREEP, cfg.STRAIGHT_CREEP)
                continue
            # weights: 0, 1000, 2000, 3000, 4000
            pos = (0*readings[0] + 1000*readings[1] + 2000*readings[2] + 3000*readings[3] + 4000*readings[4]) // total
            error = pos - cfg.LINE_CENTER
            correction = int(cfg.KP * error)

            left  = _clamp(cfg.BASE_SPEED + correction, cfg.MIN_SPD, cfg.MAX_SPD)
            right = _clamp(cfg.BASE_SPEED - correction, cfg.MIN_SPD, cfg.MAX_SPD)
            set_speeds(left, right)

        # Shorter sleep to allow rapid response when move_forward_flag is set
        time.sleep_ms(20)

def calibrate():
    """Calibrate line sensors then advance to the first intersection.

    The robot spins in place while repeatedly sampling the line sensors to
    establish min/max values.  The robot should be placed one cell behind its
    intended starting position; after calibration it drives forward to the
    first intersection and updates the global ``pos`` to ``START_POS`` so the
    caller sees that intersection as the starting point of the search. The
    metric timer begins once this intersection is reached.
    """
    global pos, move_forward_flag, METRIC_START_TIME_MS

    # 1) Spin in place to expose sensors to both edges of the line.
    #    A single full rotation is enough, so spin in one direction while
    #    repeatedly sampling the sensors.  The Pololu library recommends
    #    speeds of 920/-920 with ~10 ms pauses for calibration.
    for _ in range(50):
        if not running:
            motors_off()
            return

        set_speeds(cfg.CALIBRATE_SPEED, -cfg.CALIBRATE_SPEED)
        line_sensors.calibrate()
        time.sleep_ms(5)
        
    motors_off()
    bump.calibrate()
    time.sleep_ms(5)


    # 2) Move forward until an intersection is detected.  After the forward
    #    move the robot is sitting on our true starting cell (defined by
    #    ``START_POS`` at the top of the file) so overwrite any temporary
    #    position with that constant and mark the cell visited.
    move_forward_flag = True
    while move_forward_flag:
        uart_service()
        time.sleep_ms(1)
    pos[0], pos[1] = START_POS
    if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE:
        grid[idx(pos[0], pos[1])] = CELL_SEARCHED
    update_prob_map()
    publish_position()
    publish_visited(pos[0], pos[1])

    motors_off()
    METRIC_START_TIME_MS = time.ticks_ms()
    gc.collect()
    

def at_intersection_and_white():
    """
    Detect a 'clue':
      - Center line sensor reads white ( < cfg.MIDDLE_WHITE_THRESH )
    Returns bool.
    """
    r = line_sensors.read_calibrated()      # [0]..[4], center is [2]
    if r[2] < cfg.MIDDLE_WHITE_THRESH:
        buzz('clue')
        return True
    else:
        return False


def check_current_cell_for_clue(stage="start"):
    """Check the current cell for a clue without moving off of it."""
    global first_clue_seen, FIRST_CLUE_TIME_MS, system_clues_found, FIRST_CLUE_POSITION
    if not running or found_object:
        return
    if at_intersection_and_white():
        clue = (pos[0], pos[1])
        is_new = clue not in clues
        if is_new:
            clues.append(clue)
            system_clues_found += 1
        first_clue_seen = True
        if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
            FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
            FIRST_CLUE_POSITION = (pos[0], pos[1])
        print(f"[INFO] {stage}: clue detected at {clue}")
        publish_clue(pos[0], pos[1])
        if is_new:
            update_prob_map()
            gc.collect()


flash_LEDS(GREEN,1)
# ===========================================================
# Heading / Turning (cardinal NSEW)
# ===========================================================
def rotate_degrees(deg):
    """
    Rotate in place by a signed multiple of 90°.
    deg ∈ {-180, -90, 0, 90, 180}
    Obeys 'running' flag and always cuts motors at the end.
    """
    
    if deg == 0 or not running:
        motors_off()
        return
    
    #inch forward to make clean turn
    set_speeds(cfg.BASE_SPEED, cfg.BASE_SPEED)
    time.sleep(.2)
    motors_off()

    if deg == 180 or deg == -180:
        buzz('turn')
        set_speeds(cfg.TURN_SPEED, -cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_180_MS)

    elif deg == 90:
        buzz('turn')
        set_speeds(cfg.TURN_SPEED, -cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_90_MS)

    elif deg == -90:
        buzz('turn')
        set_speeds(-cfg.TURN_SPEED, cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_90_MS)

    motors_off()

def quarter_turns(from_dir, to_dir):
    if from_dir == to_dir:
        return 0
    if from_dir is None:
        return 1
    try:
        fi = DIRS4.index(from_dir)
        ti = DIRS4.index(to_dir)
    except ValueError:
        return 1
    delta = (ti - fi) % 4
    if delta == 2:
        return 2
    return 1

def turn_towards(cur, nxt):
    """
    Turn from current heading to face the neighbor cell `nxt`.
    - cur: (x,y) current cell
    - nxt: (x,y) next cell (must be a 4-neighbor of cur)
    Updates the global 'heading'.
    """
    global heading
    dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
    target = (dx, dy)

    i = DIRS4.index(heading)
    j = DIRS4.index(target)
    delta = (j - i) % 4

    # Map delta to minimal signed degrees
    if delta == 0:   deg = 0
    elif delta == 1: deg = 90
    elif delta == 2: deg = 180
    elif delta == 3: deg = -90

    rotate_degrees(deg)
    heading = target
flash_LEDS(GREEN,1)
# ===========================================================
# Reward Model (clues) & Pre-Clue Serpentine Bias
# ===========================================================
def update_prob_map():
    """
    Recompute prob_map.
    - Base uniform prior
    - Add Manhattan-decay bumps around all clues
    - Visited cells get zero probability
    """
    # Simple energy tracking - no function call counting needed
    total_cells = GRID_SIZE * GRID_SIZE
    has_clues = len(clues) > 0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            i = idx(x, y)
            if grid[i] == CELL_SEARCHED:  # visited
                prob_map[i] = 0.0
                continue
            if not has_clues:
                prob_map[i] = 1.0 / total_cells
                continue
            s = 0.0
            for (cx, cy) in clues:
                s += 1.0 / (1 + abs(x - cx) + abs(y - cy))
            prob_map[i] = s

def distance_from_center(coord):
    """Return the distance from the grid center along one axis.

    Used to penalize center-ward moves before the first clue is seen, without
    assigning robots to specific sides of the grid.
    """
    return abs(coord - GRID_CENTER)

def centerward_step_cost(curr_x, curr_y, next_x, next_y):
    """Pre-clue only: Penalize steps that move inward toward the center on either axis."""
    if first_clue_seen:
        return 0.0
    cost = 0.0
    if next_x != curr_x:
        d_curr = distance_from_center(curr_x)
        d_next = distance_from_center(next_x)
        if d_next < d_curr:
            cost += CENTER_STEP * (d_curr - d_next)
    if next_y != curr_y:
        d_curr = distance_from_center(curr_y)
        d_next = distance_from_center(next_y)
        if d_next < d_curr:
            cost += CENTER_STEP * (d_curr - d_next)
    return cost

def i_should_yield(ix, iy):
    """Yield if a peer reserved or currently occupies (ix, iy)."""
    # Simple energy tracking - no function call counting needed
    for pid, (px, py) in peer_intent.items():
        if (px, py) == (ix, iy):
            return True
    for pid, (px, py) in peer_pos.items():
        if (px, py) == (ix, iy):
            return True
    return False

def pick_goal():
    """Select a goal we can outbid peers for using Manhattan-distance bids."""
    # Simple energy tracking - no function call counting needed
    reserved_by_peers = {
        cell for rid, cell in peer_goal.items()
        if rid != ROBOT_ID and cell is not None
    }
    predicted_positions = {}
    for rid, cell in peer_intent.items():
        if rid != ROBOT_ID and cell is not None:
            predicted_positions[rid] = cell
    for rid, cell in peer_pos.items():
        if rid != ROBOT_ID and rid not in predicted_positions and cell is not None:
            predicted_positions[rid] = cell
    for rid, cell in peer_goal.items():
        if rid != ROBOT_ID and rid not in predicted_positions and cell is not None:
            predicted_positions[rid] = cell

    best = None
    best_val = -1e9
    fallback_best = None
    fallback_val = -1e9

    def can_win(cell, reward):
        my_bid = reward - (abs(cell[0] - pos[0]) + abs(cell[1] - pos[1]))
        for rid, start in predicted_positions.items():
            peer_bid = reward - (abs(cell[0] - start[0]) + abs(cell[1] - start[1]))
            if peer_bid > my_bid:
                return False
            if peer_bid == my_bid and rid < ROBOT_ID:
                return False
        return True

    def consider(cell):
        nonlocal best, best_val, fallback_best, fallback_val
        if cell is None:
            return
        x, y = cell
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return
        i = idx(x, y)
        if grid[i] != CELL_UNSEARCHED:
            return
        reward = prob_map[i] * REWARD_FACTOR
        if cell not in reserved_by_peers or cell == current_goal:
            if reward > fallback_val:
                fallback_val = reward
                fallback_best = cell
        if cell in reserved_by_peers:
            return
        if not can_win(cell, reward):
            return
        if reward > best_val:
            best_val = reward
            best = cell

    if heading != (0, 0):
        consider((pos[0] + heading[0], pos[1] + heading[1]))

    if best is None and heading != (0, 0):
        left = (-heading[1], heading[0])
        right = (heading[1], -heading[0])
        for sx, sy in (left, right):
            consider((pos[0] + sx, pos[1] + sy))
            if best == (pos[0] + sx, pos[1] + sy):
                break

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            consider((x, y))

    if best is not None:
        return best
    if fallback_best is not None:
        return fallback_best

    unknowns = [
        (x, y)
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
        if grid[idx(x, y)] == CELL_UNSEARCHED and (x, y) not in reserved_by_peers
    ]
    if unknowns:
        return min(unknowns, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
    return None
flash_LEDS(GREEN,1)
# ===========================================================
# A* Planner (4-neighbor grid, cardinal)
# ===========================================================
def a_star(start, goal):
    """
    A* over the 4-neighbor grid, with costs:
      +1 per step
      + TURN_COST per 90-degree heading change
      + centerward_step_cost (pre-clue serpentine)
      + cfg.VISITED_STEP_PENALTY if stepping onto a visited cell (grid==2)
      + INTENT_PENALTY if stepping into the other's reserved next cell or current position
    The reward from prob_map is applied as a bonus in the node priority.
    Returns a path as a list: [start, ..., goal], or [] if failure.
    """
    # Simple energy tracking - no function call counting needed
    frontier.clear()
    for i in range(GRID_SIZE * GRID_SIZE):
        came_from[i] = -1
        cost_so_far[i] = 1e30

    start_idx = idx(start[0], start[1])
    goal_idx = idx(goal[0], goal[1])
    heapq.heappush(frontier, (0, start_idx, heading))
    came_from[start_idx] = start_idx
    cost_so_far[start_idx] = 0.0
    turn_cost_per_turn = TURN_COST if not first_clue_seen else TURN_COST * 0.5

    while frontier and running and not found_object:
        _, current_idx, cur_dir = heapq.heappop(frontier)
        if current_idx == goal_idx:
            break

        cx = current_idx % GRID_SIZE
        cy = GRID_SIZE - 1 - (current_idx // GRID_SIZE)
        for dx, dy in DIRS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            i = idx(nx, ny)
            if grid[i] == CELL_OBSTACLE:  # obstacle/reserved
                continue

            move_cost = 1.0
            turns = quarter_turns(cur_dir, (dx, dy))
            turn_cost = turn_cost_per_turn * turns
            visited_pen = cfg.VISITED_STEP_PENALTY if grid[i] == CELL_SEARCHED else 0.0
            serp_pen = centerward_step_cost(cx, cy, nx, ny)
            base_cost = move_cost + turn_cost + visited_pen + serp_pen

            for pid, (ix, iy) in peer_intent.items():
                if (ix, iy) == (nx, ny):
                    base_cost += INTENT_PENALTY
                    break
            else:
                for pid, (px, py) in peer_pos.items():
                    if (px, py) == (nx, ny):
                        base_cost += INTENT_PENALTY
                        break

            reward_bonus = prob_map[i] * REWARD_FACTOR
            max_bonus = base_cost - 0.01
            if max_bonus < 0.0:
                max_bonus = 0.0
            if reward_bonus > max_bonus:
                reward_bonus = max_bonus

            step_cost = base_cost - reward_bonus
            if step_cost < 0.01:
                step_cost = 0.01

            new_cost = cost_so_far[current_idx] + step_cost

            if new_cost < cost_so_far[i]:
                cost_so_far[i] = new_cost
                priority = (
                    new_cost
                    + abs(goal[0] - nx)
                    + abs(goal[1] - ny)
                )
                heapq.heappush(frontier, (priority, i, (dx, dy)))
                came_from[i] = current_idx

    if came_from[goal_idx] == -1:
        return []

    # Reconstruct path
    path = []
    cur_idx = goal_idx
    while cur_idx != start_idx:
        x = cur_idx % GRID_SIZE
        y = GRID_SIZE - 1 - (cur_idx // GRID_SIZE)
        path.append((x, y))
        cur_idx = came_from[cur_idx]
    path.reverse()
    return [start] + path


flash_LEDS(GREEN,1)
# ===========================================================
# Main Search Loop
# ===========================================================
def search_loop():
    """Main mission loop.

    1. Update the probability map.
    2. Choose a goal: sweep bias before clues, reward chasing after.
    3. Plan with A* using turn, center, intent, and reward costs.
    4. Publish intent, turn, and advance one cell (abort on bump).
    5. Mark the cell, report status, and check for clues.
    6. Repeat until the object is found or no goals remain.

    Motors are always stopped in a ``finally`` block.
    """
    global first_clue_seen, move_forward_flag, start_signal, METRIC_START_TIME_MS, pos, yield_count, path_replan_count, goal_replan_count, FIRST_CLUE_TIME_MS, current_goal, system_clues_found, FIRST_CLUE_POSITION, system_visits, busy_ms, mem_free_min

    try:
        calibrate()
        update_prob_map()

        # wait for hub start command, periodically sharing start position
        last_pose_publish = time.ticks_ms()
        while not start_signal:
            uart_service()
            now = time.ticks_ms()
            if time.ticks_diff(now, last_pose_publish) >= 3000:
                publish_position()
                publish_visited(pos[0],pos[1])
                last_pose_publish = now
            time.sleep_ms(10)
        METRIC_START_TIME_MS = time.ticks_ms()
        check_current_cell_for_clue("start_signal")

        while running and not found_object:
            busy_timer_reset()
            # free any unused memory from previous iteration to avoid
            # MicroPython allocation failures during long searches
            gc.collect()
            update_mem_headroom()

            try:
                prev_goal = current_goal
                goal = pick_goal()
                if goal is None:
                    current_goal = None
                    break

                if goal != prev_goal:
                    if first_clue_seen:
                        goal_replan_count += 1
                    publish_goal(goal[0], goal[1])
                    current_goal = goal

                path = a_star(tuple(pos), goal)
                update_mem_headroom()
                # Maintain low memory usage between planning iterations
                gc.collect()
                if len(path) < 2:
                    break

                nxt = path[1]

                # Reserve the next cell so the other robot yields if it wanted the same
                publish_intent(nxt[0], nxt[1])

                # Give peers a moment to publish their intent and process it
                for _ in range(5):
                    uart_service()
                    busy_timer_pause()
                    time.sleep_ms(10)
                    busy_timer_resume()

                if i_should_yield(nxt[0], nxt[1]):
                    # Short back-off then replan
                    path_replan_count += 1
                    yield_count += 1
                    busy_timer_pause()
                    # Simple energy tracking - no function call counting needed
                    time.sleep_ms(300)
                    continue

                # Face the neighbor and try to move one cell
                busy_timer_pause()
                turn_towards(tuple(pos), nxt)
                if not running or found_object:
                    break

                move_forward_flag = True
                while move_forward_flag:
                    uart_service()
                    time.sleep_ms(1)
                busy_timer_resume()

                # Arrived + update state & publish
                pos[0], pos[1] = nxt[0], nxt[1]
                record_intersection(pos[0], pos[1])
                grid[idx(pos[0], pos[1])] = CELL_SEARCHED
                publish_position()
                publish_visited(pos[0], pos[1])

                # Clue detection: centered + white center sensor
                if at_intersection_and_white():
                    buzz('clue')
                    clue = (pos[0], pos[1])
                    if clue not in clues:
                        clues.append(clue)
                        system_clues_found += 1
                        first_clue_seen = True
                        if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                            FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                            FIRST_CLUE_POSITION = (pos[0], pos[1])
                        publish_clue(pos[0], pos[1])
                        update_prob_map()
                        update_mem_headroom()
                        gc.collect()
            finally:
                busy_ms += busy_timer_value_ms()
                update_mem_headroom()

    finally:
        motors_off()   # safety: ensure motors are cut even on exceptions
flash_LEDS(GREEN,1)
# ===========================================================
# Entry Point
# ===========================================================

flash_LEDS(RED,1)
# Start the single UART RX thread (clean exit when 'running' goes False)
_thread.start_new_thread(move_forward_one_cell, ())

# Kick off the mission
try:
    search_loop()
finally:
    # Ensure absolutely everything is stopped
    running = False
    metrics_log()
    flash_LEDS(RED,5)
    time.sleep_ms(200)  # give RX thread time to fall out cleanly


