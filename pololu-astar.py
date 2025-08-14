# ===========================================================
# Pololu 3pi+ 2040 OLED — Two-Robot Adaptive Search (UART→ESP32→MQTT bridge)
# ===========================================================
# This script runs on the Pololu 3pi+ 2040 OLED (MicroPython).
# - Transport: UART only. The ESP32 reads these lines and publishes to MQTT.
# - Topics/strings: match your integrated format (status/visited/clue/alert).
# - Behavior:
#   * Before any clue: do an outside→in "lawn-mower" sweep on each half,
#     encouraged by a higher center-ward cost than the turn cost.
#   * After first clue: switch to reward-chasing (argmax reward_map).
#   * Intent reservation: publish your next cell; avoid the other's reserved cell.
#   * Object is bump-only: on bump, publish alert and stop both robots immediately.
#   * Clues are intersections where center line sensor is white (and robot is centered).
#
# Threads:
#   * One background thread reads UART lines and updates shared state.
#   * Main thread plans/moves. Both paths guarantee motors are cut on any stop.
#
# Tuning notes:
#   * Set UART pins/baud as per Pololu board.
#   * Calibrate line sensors; adjust MIDDLE_WHITE_THRESH as needed.
#   * Adjust turn timings (YAW_90_MS/YAW_180_MS) to your platform.
# ===========================================================

'''
TO-DO
UPDATED 13AUG
-robot turns right when it should be at the end of grid.
verify that its location is corret via nano terminal
- robot doesnt fully turn left, may be hardware issue
- fix intersection debounce. temporarily at 1
-bump sensors do nothing when pressed. also view terminal to see if reading
added red flash to traoubleshoot. does not flash when pressed
- after set amount of intersection buzzer sounds and gives 5 reed flashes
indicating power off. need to attach screen to see error.
 possibly should have pololu write errors to a text file 
- pololu does not adhere to lawnmower sweep after first row
'''

import time
import _thread
import heapq
import sys
from machine import UART, Pin
from pololu_3pi_2040_robot import robot
from pololu_3pi_2040_robot.extras import editions

# -----------------------------
# Robot identity & start pose
# -----------------------------
ROBOT_ID = "00"                         
OTHER_ROBOT_ID = "01" 
GRID_SIZE = 5

# Starting position & heading (grid coordinates, cardinal heading)
# pos = (x, y)    heading = (dx, dy) where (0,-1)=N, (1,0)=E, (0,1)=S, (-1,0)=W
if ROBOT_ID == "00":
    START_POS = (0, 0) #southern bot 00 starts facing north
    START_HEADING = (0, -1)
else:
    START_POS = (GRID_SIZE -1 , GRID_SIZE -1) #northern bot, 01, starts facing south
    START_HEADING = (0, 1)

# UART0 for ESP32 communication (TX=GP28, RX=GP29)
uart = UART(0, baudrate=230400, tx=28, rx=29)

# -----------------------------
# Grid / Maps / Shared State
# -----------------------------
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 0=unknown, 1=obstacle (reserved), 2=visited
prob_map = [[1 / (GRID_SIZE * GRID_SIZE) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
reward_map = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
clues = []                            # list of (x, y) clue cells

pos = [START_POS[0], START_POS[1]]    # current grid pos
heading = (START_HEADING[0], START_HEADING[1])

# Run flags (checked by loops/threads for clean exits)
running = True                         # master run flag
found_object = False                   # set True on bump or peer alert
first_clue_seen = False                # once True, we disable lawn-mower bias

# Intent reservation from the other robot
other_intent = None                    # (x, y) or None
other_intent_time_ms = 0

# -----------------------------
# Soft split (pre-clue only)
# 00 prefers left edge; 01 prefers right edge
# We implement serpentine as "center-ward hop cost" > turning cost
# -----------------------------
PREFERS_LEFT = (ROBOT_ID == "00")  # which outer edge is "yours"

# Cost shaping (pre-clue lawn-mower / serpentine)
CENTER_STEP = 0.7        # cost per step toward the center when switching columns (must be > turn penalty ~=1)
SWITCH_COL_BASE = 0.2    # small base penalty for switching columns (pre-clue)
MIDDLE_WHITE_THRESH = 250  # center sensor threshold for "white" (tune by calibration)
# ---- Tuning knobs ----
KP = 0.5                # proportional gain around LINE_CENTER
BASE_SPEED = 900          # nominal wheel speed
MIN_SPD = 400             # clamp low (avoid stall)
MAX_SPD = 1200            # clamp high
LINE_CENTER = 2000        # weighted position target (0..4000)
BLACK_THRESH = 600        # calibrated "black" threshold (0..1000)
INTERSECTION_SAMPLES = 2  # consecutive reads to confirm intersection
STRAIGHT_CREEP = 600     # forward speed while "locked" straight
START_LOCK_MS = 500       # hold straight this long after function starts

# persistent state for debounce/lock
_intersection_hits = 0
_lock_intersection = False  # when True, ignore P-correction and drive straight

# Intent settings
INTENT_TTL_MS = 1200     # reservation lifetime
INTENT_PENALTY = 8.0     # strong penalty to avoid stepping into the other's reserved cell

# -----------------------------
# Motion tuning (line follow / turns)
# -----------------------------

TURN_SPEED = 900
YAW_90_MS = 0.15
YAW_180_MS = 0.3

# -----------------------------
# Hardware interfaces
# -----------------------------
motors = robot.Motors()
line_sensors = robot.LineSensors()
bump = robot.BumpSensors()
rgb_leds = robot.RGBLEDs()
rgb_leds.set_brightness(10)

# ===========================================================
# Utility: Motors & Stop Control
# ===========================================================

def flash_green_LEDS(num_times):
    for _ in range(num_times):
        # Turn all LEDs green
        for led in range(6):
            rgb_leds.set(led, [0, 230, 0])
        rgb_leds.show()
        time.sleep_ms(100)

        # Turn all LEDs off
        for led in range(6):
            rgb_leds.set(led, [0, 0, 0])
        rgb_leds.show()
        time.sleep_ms(100)
        
def flash_red_LEDS(num_times):
    for _ in range(num_times):
        # Turn all LEDs green
        for led in range(6):
            rgb_leds.set(led, [230, 0, 0])
        rgb_leds.show()
        time.sleep_ms(100)

        # Turn all LEDs off
        for led in range(6):
            rgb_leds.set(led, [0, 0, 0])
        rgb_leds.show()
        time.sleep_ms(100)
        
flash_green_LEDS(1)
    
def motors_off():
    """Hard stop both wheels (safety: call in finally/stop paths)."""
    motors.set_speeds(0, 0)

def stop_all(reason=""):
    """
    Idempotent global stop:
      - Set flags so all loops/threads exit
      - Ensure motors are off
      - Set a green LED to indicate finished
    """
    global running, found_object
    found_object = True
    running = False
    motors_off()

def stop_and_alert_object():
    """
    Called when THIS robot detects the object via bump.
    Publishes alert and performs a global stop.
    """
    publish_object(pos[0], pos[1])
    stop_all("object")

flash_green_LEDS(1)
# ===========================================================
# UART Messaging
# Format: "<ID>/<topic>:<payload>\n"
# Examples:
#   00/status:pos:3,4;heading:0,1
#   00/visited:visit:3,4
#   00/clue:clue:6,2
#   00/alert:object:5,7
#   00/status:intent:4,4
# ===========================================================
def uart_send(topic, payload):
    """Send a single line to ESP32; it forwards to MQTT."""
    line = f"{ROBOT_ID}/{topic}={payload}-"
    uart.write(line)

def publish_position():
    """Publish current pose (for UI/diagnostics)."""
    uart_send("status", f"pos:{pos[0]},{pos[1]};heading:{heading[0]},{heading[1]}")

def publish_visited(x, y):
    """Publish that we visited cell (x,y)."""
    uart_send("visited", f"visit:{x},{y}")

def publish_clue(x, y):
    """Publish a clue at (x,y)."""
    uart_send("clue", f"clue:{x},{y}")

def publish_object(x, y):
    """Publish that we found the object at (x,y)."""
    uart_send("alert", f"object:{x},{y}")

def publish_intent(x, y):
    """
    Publish our intended next cell (reservation).
    Other robot will penalize stepping into this cell for INTENT_TTL_MS.
    """
    uart_send("status", f"intent:{x},{y}")

def handle_uart_line(line):
    """
    Parse and apply incoming messages from the other robot.

    Accepts:
      "01/visited:visit:x,y"   → mark visited (optional mirror)
      "01/clue:clue:x,y"       → add clue, update reward
      "01/alert:object:x,y"    → global stop
      "01/status:intent:x,y"   → record reservation

    Ignores:
      - messages not from OTHER_ROBOT_ID **fix this for mmore bots
      - other status fields we don't currently need
    """
    global other_intent, other_intent_time_ms, first_clue_seen

    # Minimal parsing: "<sender>/<topic>:<payload>"
    try:
        left, payload = line.split("=", 1)
        sender, topic = left.split("/", 1)
    except ValueError:
        return
    if sender != OTHER_ROBOT_ID:
        return

    if topic == "visited" and payload.startswith("visit"):
        x, y = map(int, payload[6:].split(","))
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[y][x] == 0:
            grid[y][x] = 2

    elif topic == "clue" and payload.startswith("clue"):
        x, y = map(int, payload[5:].split(","))
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            clues.append((x, y))
            first_clue_seen = True
            update_prob_map()

    elif topic == "alert" and payload.startswith("object"):
        # Peer found the object → stop immediately
        stop_all("peer_object")

    elif topic == "status" and payload.startswith("intent"):
        ix, iy = map(int, payload[7:].split(","))
        other_intent = (ix, iy)
        other_intent_time_ms = time.ticks_ms()

def uart_rx_loop():
    """
    Background reader thread:
      - Buffers bytes until newline
      - Calls handle_uart_line() per complete line
      - Respects the 'running' flag for clean exit
    """
    buf = b""
    while running:
        if uart.any():
            b = uart.read(1)
            if not b:
                continue
            if b == b"-":
                line = buf.decode(errors="ignore").strip()
                msg = line[:-1]
                if line:
                    handle_uart_line(line)
                buf = b""
            else:
                buf += b
        else:
            time.sleep_ms(10)

# ===========================================================
# Sensing & Motion
# ===========================================================
flash_green_LEDS(1)
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def weighted_position(readings):
    """
    readings: 5 calibrated values (0..1000)
    returns: int position 0..4000, or None if no line detected
    """
    total = readings[0] + readings[1] + readings[2] + readings[3] + readings[4]
    if total == 0:
        return None
    # weights: 0, 1000, 2000, 3000, 4000
    pos = (0*readings[0] + 1000*readings[1] + 2000*readings[2]
           + 3000*readings[3] + 4000*readings[4]) // total
    return pos

def _outer_either_black(readings):
    """True if left outer OR right outer sees black (handles T-intersections)."""
    return (readings[0] >= BLACK_THRESH) or (readings[4] >= BLACK_THRESH)

def _update_intersection_debounce(readings):
    """
    Count consecutive frames where an outer sensor is black.
    Return True once INTERSECTION_SAMPLES are seen in a row.
    """
    global _intersection_hits
    if _outer_either_black(readings):
        _intersection_hits += 1
        return True ## TS temporary
    else:
        _intersection_hits = 0
    return False ## TS temporary

def bumped():
    """Return True only if a bumper is pressed continuously for ~40 ms."""
    if not (bump.left_is_pressed() or bump.right_is_pressed()):
        return False
    if bump.left_is_pressed() or bump.right_is_pressed():
        flash_red_LEDS(1)
        return True


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
    global _intersection_hits, _lock_intersection
    _intersection_hits = 0
    _lock_intersection = False

    # Initial lock to roll straight for half a second
    lock_release_time = time.ticks_add(time.ticks_ms(), START_LOCK_MS)

    while running:
        # 1) Safety/object check
        if bumped():
            stop_and_alert_object()
            motors_off()
            return False

        # 2) Read sensors
        readings = line_sensors.read_calibrated()

        # 3) During initial lock window, always drive straight
        if time.ticks_diff(time.ticks_ms(), lock_release_time) < 0:
            motors.set_speeds(STRAIGHT_CREEP, STRAIGHT_CREEP)
            continue

        # 4) Candidate intersection? lock heading immediately
        if _outer_either_black(readings) and not _lock_intersection:
            _lock_intersection = True
            _intersection_hits = 1  # count this frame

        # 5) While locked: confirm or keep rolling straight
        if _lock_intersection:
            if _update_intersection_debounce(readings):
                time.sleep(.1)
                motors_off()
                flash_green_LEDS(1)
                return True
            motors.set_speeds(STRAIGHT_CREEP, STRAIGHT_CREEP)
            continue

        # 6) Normal P-control when not locked
        total = readings[0] + readings[1] + readings[2] + readings[3] + readings[4]
        if total == 0:
            # line lost; creep straight (or call your recovery here)
            motors.set_speeds(400, 400)
            continue

        pos = weighted_position(readings)  # 0..4000 or None
        if pos is None:
            motors.set_speeds(400, 400)
            continue

        error = pos - LINE_CENTER
        correction = int(KP * error)

        left  = _clamp(BASE_SPEED + correction, MIN_SPD, MAX_SPD)
        right = _clamp(BASE_SPEED - correction, MIN_SPD, MAX_SPD)
        motors.set_speeds(left, right)

    motors_off()
    return False

def calibrate():
    """Calibrate line sensors then advance to the first intersection.

    The robot spins in place while repeatedly sampling the line sensors to
    establish min/max values.  The robot should be placed one cell behind its
    intended starting position; after calibration it drives forward to the
    first intersection and updates the global ``pos`` to ``START_POS`` so the
    caller sees that intersection as the starting point of the search.
    """
    global pos

    # 1) Spin in place to expose sensors to both edges of the line.
    #    A single full rotation is enough, so spin in one direction while
    #    repeatedly sampling the sensors.  The Pololu library recommends
    #    speeds of 920/-920 with ~10 ms pauses for calibration.
    for _ in range(50):
        if not running:
            motors_off()
            return

        motors.set_speeds(920, -920)
        line_sensors.calibrate()
        time.sleep_ms(10)

    motors_off()

    # 2) Move forward until an intersection is detected.  After the forward
    #    move the robot is sitting on our true starting cell (defined by
    #    ``START_POS`` at the top of the file) so overwrite any temporary
    #    position with that constant and mark the cell visited.
    if move_forward_one_cell():
        pos[0], pos[1] = START_POS
        if 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE:
            grid[pos[1]][pos[0]] = 2

    motors_off()
    

def at_intersection_and_white():
    """
    Detect a 'clue':
      - Center line sensor reads white ( < MIDDLE_WHITE_THRESH )
      - Position is near the center of the line (~ at an intersection)
    Returns bool.
    """
    r = line_sensors.read_calibrated()      # [0]..[4], center is [2]
    pos = weighted_position(r)
    center_white = r[2] < MIDDLE_WHITE_THRESH
    centered = abs(pos - LINE_CENTER) < 150
    return center_white and centered

flash_green_LEDS(1)
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

    if deg == 180 or deg == -180:
        motors.set_speeds(TURN_SPEED, -TURN_SPEED)
        if running: time.sleep(YAW_180_MS)

    elif deg == 90:
        motors.set_speeds(TURN_SPEED, -TURN_SPEED)
        if running: time.sleep(YAW_90_MS)

    elif deg == -90:
        motors.set_speeds(-TURN_SPEED*2, TURN_SPEED*2)
        if running: time.sleep(YAW_90_MS)

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

    dirs = [(0,-1),(1,0),(0,1),(-1,0)]   # N,E,S,W (clockwise)
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
flash_green_LEDS(1)
# ===========================================================
# Reward Model (clues) & Pre-Clue Serpentine Bias
# ===========================================================
def update_prob_map():
    """
    Recompute prob_map & reward_map.
    - Base uniform prior
    - Add Manhattan-decay bumps around all clues
    - Visited cells get zero reward
    """
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 2:  # visited
                prob_map[y][x] = 0.0
                reward_map[y][x] = 0.0
                continue
            base = 1 / (GRID_SIZE * GRID_SIZE)
            clue_sum = 0.0
            for (cx, cy) in clues:
                clue_sum += 5 / (1 + abs(x - cx) + abs(y - cy))
            prob_map[y][x] = base + clue_sum
            reward_map[y][x] = prob_map[y][x] * 5

def edge_distance_from_side(x):
    """
    Distance from "your" outer edge:
      - Robot 00: from left edge (x=0)
      - Robot 01: from right edge (x=GRID_SIZE-1)
    Used to define what 'toward center' means.
    """
    return x if PREFERS_LEFT else (GRID_SIZE - 1 - x)

def centerward_step_cost(curr_x, next_x):
    """
    Pre-clue only: Penalize stepping toward the center *more than* turning.
    - Staying in the same column costs 0 (encourages N–S sweeping).
    - Switching columns pays a base penalty.
    - If the switch moves 'inward' (toward center), add CENTER_STEP * delta.
    """
    if first_clue_seen:
        return 0.0
    if next_x == curr_x:
        return 0.0
    d_curr = edge_distance_from_side(curr_x)
    d_next = edge_distance_from_side(next_x)
    toward_center = (d_next < d_curr)
    cost = SWITCH_COL_BASE
    if toward_center:
        cost += CENTER_STEP * (d_curr - d_next)
    return cost

def is_other_intent_active():
    """True if the other's reservation is still fresh."""
    if other_intent is None:
        return False
    return time.ticks_diff(time.ticks_ms(), other_intent_time_ms) <= INTENT_TTL_MS

def i_should_yield(ix, iy):
    """
    Deterministic back-off on intent collision.
    Lower ID yields if both reserve the same cell (rare but possible).
    """
    return (other_intent == (ix, iy)) and (ROBOT_ID < OTHER_ROBOT_ID)

def pick_goal():
    """
    Choose a goal cell:
      - Post-clue: pure argmax(reward_map) among unknown cells.
      - Pre-clue: argmax(reward) but statically biased against center
                  via edge-distance (keeps goals in outer strips first).
    Fallback: nearest unknown if all rewards are flat.
    """
    best = None
    best_val = -1e9
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] != 0:
                continue
            val = reward_map[y][x]
            if not first_clue_seen:
                # Static nudge to keep targets in outer strips pre-clue
                # (dynamic step cost in A* does the heavy lifting)
                val -= 0.3 * edge_distance_from_side(x)
            if val > best_val:
                best_val = val
                best = (x, y)

    if best is None:
        # Fallback: nearest unknown
        unknowns = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if grid[y][x] == 0]
        if unknowns:
            best = min(unknowns, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
    return best
flash_green_LEDS(1)
# ===========================================================
# A* Planner (4-neighbor grid, cardinal)
# ===========================================================
def a_star(start, goal):
    """
    A* over the 4-neighbor grid, with costs:
      +1 per step
      +1 turn penalty if direction changes
      + centerward_step_cost (pre-clue serpentine)
      + INTENT_PENALTY if stepping into the other's reserved next cell
      - reward_map (seek high-reward cells)
    Returns a path as a list: [start, ..., goal], or [] if failure.
    """
    frontier = [(0, start, heading)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier and running and not found_object:
        _, current, cur_dir = heapq.heappop(frontier)
        if current == goal:
            break

        cx, cy = current
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if grid[ny][nx] == 1:  # reserved for obstacles (not used yet)
                continue

            new_cost = cost_so_far[current] + 1

            # Turning penalty
            if (dx, dy) != cur_dir:
                new_cost += 1

            # Reward shaping (prefer high reward)
            new_cost -= reward_map[ny][nx]

            # Pre-clue: penalize inward hops (serpentine)
            new_cost += centerward_step_cost(cx, nx)

            # Reservation: avoid other's intended next cell
            if is_other_intent_active() and (nx, ny) == other_intent:
                new_cost += INTENT_PENALTY

            nxt = (nx, ny)
            if (nxt not in cost_so_far) or (new_cost < cost_so_far[nxt]):
                cost_so_far[nxt] = new_cost
                priority = new_cost + abs(goal[0] - nx) + abs(goal[1] - ny)
                heapq.heappush(frontier, (priority, nxt, (dx, dy)))
                came_from[nxt] = current

    if goal not in came_from:
        return []
    # Reconstruct path
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return [start] + path

flash_green_LEDS(1)
# ===========================================================
# Main Search Loop
# ===========================================================
def search_loop():
    """
    High-level mission loop:
      1) Update reward map
      2) Pick a goal (pre-clue sweep bias or post-clue reward chase)
      3) Plan with A* (costs include turn, center-ward, intent, reward)
      4) Publish intent → turn → advance one cell (with bump abort)
      5) Mark visited, publish status, detect clue (intersection/white)
      6) Repeat until object found or no goals remain
    Always cuts motors in a finally block.
    """
    global first_clue_seen

    try:
        calibrate()
        update_prob_map()
        publish_position()
        publish_visited(pos[0], pos[1])
        
        while running and not found_object:
            goal = pick_goal()
            if goal is None:
                break

            path = a_star(tuple(pos), goal)
            if len(path) < 2:
                break

            nxt = path[1]

            # Reserve the next cell so the other robot yields if it wanted the same
            publish_intent(nxt[0], nxt[1])
            if i_should_yield(nxt[0], nxt[1]):
                # Short back-off then replan
                time.sleep_ms(300)
                continue

            # Face the neighbor and try to move one cell
            turn_towards(tuple(pos), nxt)
            if not running or found_object:
                break

            if not move_forward_one_cell():
                break  # Bump or stop condition handled inside

            # Arrived → update state & publish
            pos[0], pos[1] = nxt[0], nxt[1]
            grid[pos[1]][pos[0]] = 2
            publish_position()
            publish_visited(pos[0], pos[1])

            # Clue detection: centered + white center sensor
            if at_intersection_and_white():
                clues.append((pos[0], pos[1]))
                first_clue_seen = True
                publish_clue(pos[0], pos[1])
                update_prob_map()

    finally:
        motors_off()   # safety: ensure motors are cut even on exceptions
flash_green_LEDS(1)
# ===========================================================
# Entry Point
# ===========================================================

flash_red_LEDS(5)
# Start the single UART RX thread (clean exit when 'running' goes False)
_thread.start_new_thread(uart_rx_loop, ())

# Kick off the mission
try:
    search_loop()
finally:
    # Ensure absolutely everything is stopped
    stop_all("finally")
    flash_red_LEDS(5)
    time.sleep_ms(200)  # give RX thread time to fall out cleanly
