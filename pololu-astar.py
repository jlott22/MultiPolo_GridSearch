# ===========================================================
# Pololu 3pi+ 2040 OLED — Two-Robot Adaptive Search (UART→ESP32→MQTT bridge)
# ===========================================================
# This script runs on the Pololu 3pi+ 2040 OLED (MicroPython).
# - Transport: UART only. The ESP32 reads these lines and publishes to MQTT.
# - Topics/strings: match your integrated format (status/visited/clue/alert).
# - Behavior:
#   * Before any clue: do an outside→in "lawn-mower" sweep on each half,
#     encouraged by a higher center-ward cost than the turn cost.
#   * After first clue: switch to reward-chasing (argmax derived from prob_map).
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
#   * Calibrate line sensors; adjust cfg.MIDDLE_WHITE_THRESH as needed.
#   * Adjust turn timings (cfg.YAW_90_MS/cfg.YAW_180_MS) to your platform.
# ===========================================================

'''
TO-DO
UPDATED 17AUG
- figure out what current psotion does and why needed. 
probably should be wieghted like intent. 
-test bump sensors
'''

import time
import _thread
import heapq
import sys
import gc
from array import array
from machine import UART, Pin
from pololu_3pi_2040_robot import robot
from pololu_3pi_2040_robot.extras import editions

# -----------------------------
# Robot identity & start pose
# -----------------------------
ROBOT_ID = "01"                         
OTHER_ROBOT_ID = "00" 
GRID_SIZE = 5

# Starting position & heading (grid coordinates, cardinal heading)
# pos = (x, y)    heading = (dx, dy) where (0,1)=N, (1,0)=E, (0,-1)=S, (-1,0)=W
if ROBOT_ID == "00":
    START_POS = (0, 0)  # southern bot 00 starts facing north
    START_HEADING = (0, 1)
else:
    START_POS = (GRID_SIZE -1 , GRID_SIZE -1)  # northern bot, 01, starts facing south
    START_HEADING = (0, -1)

# UART0 for ESP32 communication (TX=GP28, RX=GP29)
uart = UART(0, baudrate=115200, tx=28, rx=29)

# -----------------------------
# Grid / Maps / Shared State
# -----------------------------
grid = bytearray(GRID_SIZE * GRID_SIZE)  # 0=unknown, 1=obstacle (reserved), 2=visited
prob_map = array('f', [1 / (GRID_SIZE * GRID_SIZE)] * (GRID_SIZE * GRID_SIZE))
REWARD_FACTOR = 5
clues = []                            # list of (x, y) clue cells


def idx(x, y):
    """Convert Cartesian (x, y) to linear index in map arrays."""
    return (GRID_SIZE - 1 - y) * GRID_SIZE + x


pos = [START_POS[0], START_POS[1]]    # current grid pos
heading = (START_HEADING[0], START_HEADING[1])

# Run flags (checked by loops/threads for clean exits)
running = True                         # master run flag
found_object = False                   # set True on bump or peer alert
first_clue_seen = False                # once True, we disable lawn-mower bias
move_forward_flag = False

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
CENTER_STEP = 0.4        # cost per step toward the center when switching columns (must be > turn penalty ~=1)
SWITCH_COL_BASE = 0.3    # small base penalty for switching columns (pre-clue)

# -----------------------------
# Motion configuration
# -----------------------------
class MotionConfig:
    def __init__(self):
        self.MIDDLE_WHITE_THRESH = 800  # center sensor threshold for "white" (tune by calibration)
        self.VISITED_STEP_PENALTY = 1.2
        self.KP = 0.5                # proportional gain around LINE_CENTER
        self.CALIBRATE_SPEED = 1130  # speed to rotate when calibrating
        self.BASE_SPEED = 800        # nominal wheel speed
        self.MIN_SPD = 400           # clamp low (avoid stall)
        self.MAX_SPD = 1200          # clamp high
        self.LINE_CENTER = 2000      # weighted position target (0..4000)
        self.BLACK_THRESH = 600      # calibrated "black" threshold (0..1000)
        self.STRAIGHT_CREEP = 600    # forward speed while "locked" straight
        self.START_LOCK_MS = 500     # hold straight this long after function starts
        self.TURN_SPEED = 1000
        self.YAW_90_MS = 0.3
        self.YAW_180_MS = 0.6

cfg = MotionConfig()

# Intent settings
INTENT_TTL_MS = 1200     # reservation lifetime
INTENT_PENALTY = 8.0     # strong penalty to avoid stepping into the other's reserved cell

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

        
flash_LEDS(GREEN,1)
    
def motors_off():
    """Hard stop both wheels (safety: call in finally/stop paths)."""
    motors.set_speeds(0, 0)

def stop_all():
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
    stop_all()

flash_LEDS(GREEN,1)
# ===========================================================
# UART Messaging
# Format: "<topic#>:<payload>\n"
# position = 1, visited = 2, clue = 3, alert = 4, intent = 5
# Examples:
#   0013,4;0,1- robot 00 status update position (3,4), heading north
#   00365-
# ===========================================================
def uart_send(topic, payload):
    """Send a single line to ESP32; it forwards to MQTT."""
    line = f"{topic}.{payload}-"
    uart.write(line)

def publish_position():
    """Publish current pose (for UI/diagnostics)."""
    uart_send('1', f"{pos[0]},{pos[1]};{heading[0]},{heading[1]}")

def publish_visited(x, y):
    """Publish that we visited cell (x,y)."""
    uart_send('2', f"{x},{y}")

def publish_clue(x, y):
    """Publish a clue at (x,y)."""
    uart_send('3', f"{x},{y}")

def publish_object(x, y):
    """Publish that we found the object at (x,y)."""
    uart_send('4', f"{x},{y}")

def publish_intent(x, y):
    """
    Publish our intended next cell (reservation).
    Other robot will penalize stepping into this cell for INTENT_TTL_MS.
    """
    uart_send('5', f"{x},{y}")

def handle_msg(line):
    """
    Parse and apply incoming messages from the other robot.

    Accepts:
    011.3,4;0,1-   # topic 1: position+heading
    002.3,4-       # topic 2: visited
    003.5,2-       # topic 3: clue
    004.6,1-       # topic 4: object/alert
    005.7,2-       # topic 5: intent

    Ignores:
      - messages not from OTHER_ROBOT_ID **fix this for mmore bots
      - other status fields we don't currently need
    """
    global other_intent, other_intent_time_ms, first_clue_seen

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
            if grid[i] == 0:
                grid[i] = 2
                print('visited updated')

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
                update_prob_map()
                gc.collect()
                print('clue updated')

    elif topic == "4": #object
        # Peer found the object → stop immediately
        stop_all()
        print('object updated')

    elif topic == "1": #position, heading
        if ";" not in payload:
            return
        other_location, other_heading = payload.split(";")
        print('recivded position')

    elif topic == "5": #intent
        try:
            ix, iy = map(int, payload.split(","))
        except ValueError:
            return
        other_intent = (ix, iy)
        other_intent_time_ms = time.ticks_ms()
        print('intent processed')

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
    print('Data Read: ', data)
    for b in data:         # iterate over bytes
        rb_put_byte(b)
    while True:
        msg = rb_pull_into_msg()
        if msg is None:
            break
        print('msg: ', msg)
        handle_msg(msg)

# ===========================================================
# Sensing & Motion
# ===========================================================
flash_LEDS(GREEN,1)
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v
'''
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

def intersection_check(readings):
    """True if left outer OR right outer sees black (handles T-intersections)."""
    return (readings[0] >= cfg.BLACK_THRESH) or (readings[4] >= cfg.BLACK_THRESH)

def bumped():
    """Return True only if a bumper is pressed """
    bump.read()
    if bump.left_is_pressed() or bump.right_is_pressed():
        return True
    else:
        return False
'''

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
    global _intersection_hits, move_forward_flag
    _intersection_hits = 0
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
                motors.set_speeds(cfg.STRAIGHT_CREEP, cfg.STRAIGHT_CREEP)
                continue
            
            # 2) Read sensors
            readings = line_sensors.read_calibrated()
            
            bump.read()
            if bump.left_is_pressed() or bump.right_is_pressed():
                stop_and_alert_object()
                motors_off()
                move_forward_flag = False
                break
            
            # 4) Candidate intersection? lock heading immediately
            if readings[0] >= cfg.BLACK_THRESH or readings[4] >= cfg.BLACK_THRESH:
                motors_off()
                flash_LEDS(GREEN,1)
                move_forward_flag = False
                first_loop = True
                break

            # 6) Normal P-control when not locked
            total = readings[0] + readings[1] + readings[2] + readings[3] + readings[4]
            if total == 0:
                motors.set_speeds(cfg.STRAIGHT_CREEP, cfg.STRAIGHT_CREEP)
                continue
            # weights: 0, 1000, 2000, 3000, 4000
            pos = (0*readings[0] + 1000*readings[1] + 2000*readings[2] + 3000*readings[3] + 4000*readings[4]) // total
            error = pos - cfg.LINE_CENTER
            correction = int(cfg.KP * error)

            left  = _clamp(cfg.BASE_SPEED + correction, cfg.MIN_SPD, cfg.MAX_SPD)
            right = _clamp(cfg.BASE_SPEED - correction, cfg.MIN_SPD, cfg.MAX_SPD)
            motors.set_speeds(left, right)

        time.sleep_ms(300)

def calibrate():
    """Calibrate line sensors then advance to the first intersection.

    The robot spins in place while repeatedly sampling the line sensors to
    establish min/max values.  The robot should be placed one cell behind its
    intended starting position; after calibration it drives forward to the
    first intersection and updates the global ``pos`` to ``START_POS`` so the
    caller sees that intersection as the starting point of the search.
    """
    global pos, move_forward_flag

    # 1) Spin in place to expose sensors to both edges of the line.
    #    A single full rotation is enough, so spin in one direction while
    #    repeatedly sampling the sensors.  The Pololu library recommends
    #    speeds of 920/-920 with ~10 ms pauses for calibration.
    for _ in range(50):
        if not running:
            motors_off()
            return

        motors.set_speeds(cfg.CALIBRATE_SPEED, -cfg.CALIBRATE_SPEED)
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
        grid[idx(pos[0], pos[1])] = 2

    motors_off()
    

def at_intersection_and_white():
    """
    Detect a 'clue':
      - Center line sensor reads white ( < cfg.MIDDLE_WHITE_THRESH )
    Returns bool.
    """
    r = line_sensors.read_calibrated()      # [0]..[4], center is [2]
    if r[2] < cfg.MIDDLE_WHITE_THRESH:
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
    #inch forward to make clean turn
    motors.set_speeds(cfg.BASE_SPEED, cfg.BASE_SPEED)
    time.sleep(.2)
    motors_off()
    
    if deg == 0 or not running:
        motors_off()
        return

    if deg == 180 or deg == -180:
        motors.set_speeds(cfg.TURN_SPEED, -cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_180_MS)

    elif deg == 90:
        motors.set_speeds(cfg.TURN_SPEED, -cfg.TURN_SPEED)
        if running: time.sleep(cfg.YAW_90_MS)

    elif deg == -90:
        motors.set_speeds(-cfg.TURN_SPEED, cfg.TURN_SPEED)
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
            if grid[i] == 2:  # visited
                prob_map[i] = 0.0
                continue
          
            base = 1 / (GRID_SIZE * GRID_SIZE)
            clue_sum = 0.0
            for (cx, cy) in clues:
                clue_sum += 5 / (1 + abs(x - cx) + abs(y - cy))
            prob_map[i] = base + clue_sum


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
      - Post-clue: pure argmax(reward) among unknown cells, where
        reward = prob_map * REWARD_FACTOR.
      - Pre-clue: argmax(reward) but statically biased against center
                  via edge-distance (keeps goals in outer strips first).
    Fallback: nearest unknown if all rewards are flat.
    """
    best = None
    best_val = -1e9
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            i = idx(x, y)
            if grid[i] != 0:
                continue
            val = reward_map[i] * REWARD_FACTOR

            if not first_clue_seen:
                # Static nudge to keep targets in outer strips pre-clue
                # (dynamic step cost in A* does the heavy lifting)
                val -= 0.3 * edge_distance_from_side(x)
            if val > best_val:
                best_val = val
                best = (x, y)

    if best is None:
        # Fallback: nearest unknown
        unknowns = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if grid[idx(x, y)] == 0]
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
      + INTENT_PENALTY if stepping into the other's reserved next cell
      - prob_map * REWARD_FACTOR (seek high-reward cells)
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
            i = idx(nx, ny)
            if grid[i] == 1:  # 1 = obstacle/reserved
                continue

            new_cost = cost_so_far[current] + 1

            # Turning penalty
            if (dx, dy) != cur_dir:
                new_cost += 1

            # 🔹 Penalty for retracing visited cell
            if grid[i] == 2:   # 2 = visited
                new_cost += cfg.VISITED_STEP_PENALTY

            # Reward shaping (prefer high reward)
            new_cost -= reward_map[i] * REWARD_FACTOR

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


flash_LEDS(GREEN,1)
# ===========================================================
# Main Search Loop
# ===========================================================
def search_loop():
    """
    High-level mission loop:
      1) Update probability map
      2) Pick a goal (pre-clue sweep bias or post-clue reward chase)
      3) Plan with A* (costs include turn, center-ward, intent, reward)
      4) Publish intent → turn → advance one cell (with bump abort)
      5) Mark visited, publish status, detect clue (intersection/white)
      6) Repeat until object found or no goals remain
    Always cuts motors in a finally block.
    """
    global first_clue_seen, move_forward_flag

    try:
        calibrate()
        update_prob_map()
        publish_position()
        publish_visited(pos[0], pos[1])
        
        while running and not found_object:
            # free any unused memory from previous iteration to avoid
            # MicroPython allocation failures during long searches
            gc.collect()

            goal = pick_goal()
            if goal is None:
                break

            path = a_star(tuple(pos), goal)
            # a_star allocates several temporary structures; collect to free them
            gc.collect()
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
            
            move_forward_flag = True
            while move_forward_flag:
                uart_service()
                time.sleep_ms(1)

            # Arrived → update state & publish
            pos[0], pos[1] = nxt[0], nxt[1]
            grid[idx(pos[0], pos[1])] = 2
            publish_position()
            publish_visited(pos[0], pos[1])

            # Clue detection: centered + white center sensor
            if at_intersection_and_white():
                clue = (pos[0], pos[1])
                if clue not in clues:
                    clues.append(clue)
                    first_clue_seen = True
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
