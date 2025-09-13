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
#   * A background UART reader keeps shared state updated.
#   * The main thread plans paths and moves the robot, always stopping the
#     motors if the program exits unexpectedly.
#
# Tuning hints:
#   * Set UART pins and baud rate to match the hardware.
#   * Calibrate line sensors and adjust cfg.MIDDLE_WHITE_THRESH accordingly.
#   * Tune yaw timings (cfg.YAW_90_MS / cfg.YAW_180_MS) for your platform.
# ===========================================================

import time
import _thread
import heapq
import sys
import gc
import csv
from array import array
from machine import UART, Pin
from pololu_3pi_2040_robot import robot
from pololu_3pi_2040_robot.extras import editions
from pololu_3pi_2040_robot.buzzer import Buzzer

# TODO:
# - Ensure object reporting uses the updated location before broadcasting.
# - Strengthen intent handling to prevent collisions.
# - Consider enlarging the grid and adding a desktop logging tool.
# -----------------------------
# Robot identity & start pose
# -----------------------------
ROBOT_ID = "00"  # set to "00", "01", "02", or "03" at deployment
GRID_SIZE = 10
GRID_CENTER = (GRID_SIZE - 1) / 2

DEBUG_LOG_FILE = "debug-log.txt"

METRICS_LOG_FILE = "metrics-log.txt"
BOOT_TIME_MS = time.ticks_ms()
METRIC_START_TIME_MS = None  # set after first post-calibration intersection
start_signal = False  # set when hub command received
intersection_visits = {}
system_visits = {}  # track visits from all robots
intersection_count = 0          # steps taken by this robot
repeat_intersection_count = 0   # this robot's revisits
system_repeat_count = 0         # revisits across the whole system
yield_count = 0                 # times this robot yielded an intended move
FIRST_CLUE_TIME_MS = None       # ms from start to first clue (system-wide)
OBJECT_TIME_MS = None           # ms from start to object detection
OBJECT_STEP_COUNT = None        # intersections traversed when object was found
object_location = None  # set when object is found

buzzer = None  # replaced after hardware initialization

# Energy/Time metrics
motor_time_ms = 0              # cumulative ms motors were commanded non-zero
_motor_start_ms = None         # internal tracker for motor activity


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
    """Track intersection visits and repeated counts."""
    safe_assert(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE, "intersection out of range")
    global intersection_count, repeat_intersection_count, system_repeat_count
    intersection_count += 1
    key = (x, y)
    if key in intersection_visits:
        repeat_intersection_count += 1
        intersection_visits[key] += 1
    else:
        intersection_visits[key] = 1

    # Update system-wide visit counts
    if key in system_visits:
        system_visits[key] += 1
        system_repeat_count += 1
    else:
        system_visits[key] = 1


def metrics_log():
    """Write summary metrics for the search run and return them."""
    start = METRIC_START_TIME_MS if METRIC_START_TIME_MS is not None else BOOT_TIME_MS
    now = time.ticks_ms()
    elapsed = time.ticks_diff(now, start)
    unique_cells = len(intersection_visits)
    path_eff = (
        unique_cells / intersection_count if intersection_count else 0.0
    )
    compute_time = elapsed - motor_time_ms
    if object_location is not None and OBJECT_STEP_COUNT:
        optimal_steps = abs(object_location[0] - START_POS[0]) + abs(object_location[1] - START_POS[1])
        obj_path_eff = optimal_steps / OBJECT_STEP_COUNT if OBJECT_STEP_COUNT else 0.0
    else:
        obj_path_eff = -1.0

    metrics = {
        "elapsed_ms": elapsed,
        "compute_ms": compute_time,
        "motor_ms": motor_time_ms,
        "first_clue_ms": FIRST_CLUE_TIME_MS if FIRST_CLUE_TIME_MS is not None else -1,
        "object_ms": OBJECT_TIME_MS if OBJECT_TIME_MS is not None else -1,
        "unique_cells": unique_cells,
        "steps": intersection_count,
        "individual_revisits": repeat_intersection_count,
        "system_revisits": system_repeat_count,
        "yields": yield_count,
        "path_eff": round(path_eff, 2),
        "obj_path_eff": round(obj_path_eff, 2),
        "object": object_location,
        "clues": clues,
    }

    fieldnames = [
        "elapsed_ms",
        "compute_ms",
        "motor_ms",
        "first_clue_ms",
        "object_ms",
        "unique_cells",
        "steps",
        "individual_revisits",
        "system_revisits",
        "yields",
        "path_eff",
        "obj_path_eff",
        "object",
        "clues",
    ]

    try:
        try:
            with open(METRICS_LOG_FILE) as _fp:
                write_header = _fp.read(1) == ""
        except OSError:
            write_header = True
        with open(METRICS_LOG_FILE, "a", newline="") as _fp:
            writer = csv.DictWriter(_fp, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
    except OSError:
        pass
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

# -----------------------------
# Cost shaping for early sweeping pattern
# A small cost per step toward the center keeps robots sweeping their region
# before clues are discovered. It must exceed the turn penalty (~1).
CENTER_STEP = 0.4

# -----------------------------
# Motion configuration
# -----------------------------
class MotionConfig:
    def __init__(self):
        self.MIDDLE_WHITE_THRESH = 200  # center sensor threshold for "white" (tune by calibration) 
        self.VISITED_STEP_PENALTY = 1.2
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
    global _motor_start_ms, motor_time_ms
    if left != 0 or right != 0:
        if _motor_start_ms is None:
            _motor_start_ms = time.ticks_ms()
    else:
        if _motor_start_ms is not None:
            motor_time_ms += time.ticks_diff(time.ticks_ms(), _motor_start_ms)
            _motor_start_ms = None
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
    global object_location, found_object, OBJECT_TIME_MS, OBJECT_STEP_COUNT
    next_x = pos[0] + heading[0]
    next_y = pos[1] + heading[1]
    object_location = (next_x, next_y)
    publish_object(next_x, next_y)
    buzz('object')
    found_object = True
    if OBJECT_TIME_MS is None and METRIC_START_TIME_MS is not None:
        OBJECT_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
    if OBJECT_STEP_COUNT is None:
        OBJECT_STEP_COUNT = intersection_count
    stop_all()
    flash_LEDS(BLUE, 1)

flash_LEDS(GREEN,1)
# ===========================================================
# UART Messaging
# Format: "<topic#>:<payload>\n"
# position = 1, visited = 2, clue = 3, alert = 4, intent = 5, result = 6
# Examples:
#   0013,4;0,1- robot 00 status update position (3,4), heading north
#   00365-
# ===========================================================
def uart_send(topic, payload_len):
    """Send the prepared message in tx_buf with topic and payload_len."""
    tx_buf[0] = ord(topic)
    tx_buf[1] = ord('.')
    tx_buf[payload_len + 2] = ord('-')
    uart.write(tx_buf[:payload_len + 3])

def publish_position():
    """Publish current pose (for UI/diagnostics)."""
    i = 2
    i = _write_int(tx_buf, i, pos[0])
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, pos[1])
    tx_buf[i] = ord(';'); i += 1
    i = _write_int(tx_buf, i, heading[0])
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, heading[1])
    uart_send('1', i - 2)

def publish_visited(x, y):
    """Publish that we visited cell (x,y)."""
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('2', i - 2)

def publish_clue(x, y):
    """Publish a clue at (x,y)."""
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('3', i - 2)

def publish_object(x, y):
    """Publish that we found the object at (x,y)."""
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
    i = 2
    i = _write_int(tx_buf, i, x)
    tx_buf[i] = ord(','); i += 1
    i = _write_int(tx_buf, i, y)
    uart_send('5', i - 2)

def publish_result(msg):
    """Publish final search metrics or result to the hub."""
    # ``msg`` can be numeric; ensure it is converted to string before
    # concatenation to avoid ``TypeError: can't convert 'int' object to str``.
    uart.write("6." + str(msg) + "-")

def handle_msg(line):
    """
    Parse and apply incoming messages from the other robot or hub.

    Accepts:
    011.3,4;0,1-   # topic 1: position+heading
    002.3,4-       # topic 2: visited
    003.5,2-       # topic 3: clue
    004.6,1-       # topic 4: object/alert
    005.7,2-       # topic 5: intent
    996.1-         # topic 6: hub command

    Ignores:
      - other status fields we don't currently need
    """
    global peer_intent, peer_pos, first_clue_seen, object_location, start_signal, found_object, system_visits, system_repeat_count, FIRST_CLUE_TIME_MS, OBJECT_TIME_MS, OBJECT_STEP_COUNT

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
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            i = idx(x, y)
            grid[i] = CELL_SEARCHED
            prob_map[i] = 0.0
            if (x, y) not in intersection_visits:
                intersection_visits[(x, y)] = 1
            key = (x, y)
            if key in system_visits:
                system_visits[key] += 1
                system_repeat_count += 1
            else:
                system_visits[key] = 1
            debug_log('visited updated:', i)

    elif topic == "3":   #clue
        try:
            x, y = map(int, payload.split(","))
        except ValueError:
            return
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            clue = (x, y)
            if clue not in clues:
                clues.append(clue)
                first_clue_seen = True
                if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                    FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                update_prob_map()
                gc.collect()

    elif topic == "4": #object
        # Peer found the object → stop immediately
        try:
            x, y = map(int, payload.split(","))
            object_location = (x, y)
        except ValueError:
            object_location = None
        found_object = True
        if OBJECT_TIME_MS is None and METRIC_START_TIME_MS is not None:
            OBJECT_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
        if OBJECT_STEP_COUNT is None:
            OBJECT_STEP_COUNT = intersection_count
        stop_all()

    elif topic == "1": #position, heading
        if ";" not in payload:
            return
        other_location, other_heading = payload.split(";")
        try:
            ox, oy = map(int, other_location.split(","))
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
    time.sleep(.15)
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

    dirs = [(0,1),(1,0),(0,-1),(-1,0)]   # N,E,S,W (clockwise)
    i = dirs.index(heading)
    j = dirs.index(target)
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
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            i = idx(x, y)
            if grid[i] == CELL_SEARCHED:  # visited
                prob_map[i] = 0.0
                continue
          
            base = 1 / (GRID_SIZE * GRID_SIZE)
            clue_sum = 0.0
            for (cx, cy) in clues:
                clue_sum += 5 / (1 + abs(x - cx) + abs(y - cy))
            prob_map[i] = base + clue_sum


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
    for pid, (px, py) in peer_intent.items():
        if (px, py) == (ix, iy):
            return True
    for pid, (px, py) in peer_pos.items():
        if (px, py) == (ix, iy):
            return True
    return False

def pick_goal():
    """
    Choose a goal cell as the argmax(reward) among unknown cells, where
    reward = prob_map * REWARD_FACTOR.  The dynamic step cost in A* handles
    any pre-clue bias; no static center bias is applied here.
    Fallback: nearest unknown if all rewards are flat.
    """
    best = None
    best_val = -1e9

    # Prefer the cell straight ahead when its reward ties with others.
    fx, fy = pos[0] + heading[0], pos[1] + heading[1]
    if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
        i = idx(fx, fy)
        if grid[i] == CELL_UNSEARCHED:
            best = (fx, fy)
            best_val = prob_map[i] * REWARD_FACTOR

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            i = idx(x, y)
            if grid[i] != CELL_UNSEARCHED:
                continue
            val = prob_map[i] * REWARD_FACTOR
            if val > best_val:
                best_val = val
                best = (x, y)

    if best is None:
        # Fallback: nearest unknown
        unknowns = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if grid[idx(x, y)] == CELL_UNSEARCHED]
        if unknowns:
            best = min(unknowns, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
    return best
flash_LEDS(GREEN,1)
# ===========================================================
# A* Planner (4-neighbor grid, cardinal)
# ===========================================================
def a_star(start, goal):
    """
    A* over the 4-neighbor grid, with costs:
      +1 per step
      +1 turn penalty if direction changes
      + centerward_step_cost (pre-clue serpentine)
      + cfg.VISITED_STEP_PENALTY if stepping onto a visited cell (grid==2)
      + INTENT_PENALTY if stepping into the other's reserved next cell or current position
    The reward from prob_map is applied as a bonus in the node priority.
    Returns a path as a list: [start, ..., goal], or [] if failure.
    """
    frontier.clear()
    for i in range(GRID_SIZE * GRID_SIZE):
        came_from[i] = -1
        cost_so_far[i] = 1e30

    start_idx = idx(start[0], start[1])
    goal_idx = idx(goal[0], goal[1])
    heapq.heappush(frontier, (0, start_idx, heading))
    came_from[start_idx] = start_idx
    cost_so_far[start_idx] = 0.0

    while frontier and running and not found_object:
        _, current_idx, cur_dir = heapq.heappop(frontier)
        if current_idx == goal_idx:
            break

        cx = current_idx % GRID_SIZE
        cy = GRID_SIZE - 1 - (current_idx // GRID_SIZE)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            i = idx(nx, ny)
            if grid[i] == CELL_OBSTACLE:  # obstacle/reserved
                continue

            new_cost = cost_so_far[current_idx] + 1

            # Turning penalty
            if (dx, dy) != cur_dir:
                new_cost += 1

            # 🔹 Penalty for retracing visited cell
            if grid[i] == CELL_SEARCHED:   # visited
                new_cost += cfg.VISITED_STEP_PENALTY

            # Pre-clue: penalize inward hops (serpentine)
            new_cost += centerward_step_cost(cx, cy, nx, ny)

            # Reservation: avoid peers' intended next cells and current positions
            for pid, (ix, iy) in peer_intent.items():
                if (ix, iy) == (nx, ny):
                    new_cost += INTENT_PENALTY
                    break
            else:
                for pid, (px, py) in peer_pos.items():
                    if (px, py) == (nx, ny):
                        new_cost += INTENT_PENALTY
                        break

            if new_cost < cost_so_far[i]:
                cost_so_far[i] = new_cost
                priority = (
                    new_cost
                    + abs(goal[0] - nx)
                    + abs(goal[1] - ny)
                    - prob_map[i] * REWARD_FACTOR
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
    global first_clue_seen, move_forward_flag, start_signal, METRIC_START_TIME_MS, pos, yield_count, FIRST_CLUE_TIME_MS

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
        
        while running and not found_object:
            # free any unused memory from previous iteration to avoid
            # MicroPython allocation failures during long searches
            gc.collect()

            goal = pick_goal()
            if goal is None:
                break

            path = a_star(tuple(pos), goal)
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
                time.sleep_ms(10)

            if i_should_yield(nxt[0], nxt[1]):
                # Short back-off then replan
                yield_count += 1
                time.sleep_ms(300)
                continue

            # Face the neighbor and try to move one cell
            turn_towards(tuple(pos), nxt)
            if not running or found_object:
                break
            
            move_forward_flag = True
            while move_forward_flag:
                uart_service()
                time.sleep_ms(1)

            # Arrived → update state & publish
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
                    first_clue_seen = True
                    if FIRST_CLUE_TIME_MS is None and METRIC_START_TIME_MS is not None:
                        FIRST_CLUE_TIME_MS = time.ticks_diff(time.ticks_ms(), METRIC_START_TIME_MS)
                    publish_clue(pos[0], pos[1])
                    update_prob_map()
                    # releasing temporaries created during map update
                    gc.collect()

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
    stop_all()
    flash_LEDS(RED,5)
    time.sleep_ms(200)  # give RX thread time to fall out cleanly
