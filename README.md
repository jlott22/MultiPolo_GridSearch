# Pololu MQTT Search

This repository contains MicroPython programs and ESP32 sketches for coordinating Pololu 3pi+ 2040 robots through MQTT. An ESP32 acts as a serial-to-MQTT bridge for one or more robots.

## Overview

This project coordinates one or more Pololu 3pi+ 2040 robots using an ESP32 as a serial-to-MQTT bridge. Several robot strategies are available:

- **`pololu-astar.py`** – A* search with probability maps and intent reservations.
- **`pololu-astar-reservation.py`** – A* search that also publishes goal reservations.
- **`pololu-nextcell.py`** – Probabilistic neighbor selection without full path planning.
- **`pololu-sweep.py`** – Hardcoded lawn‑mower sweep path.

Companion ESP32 sketches translate the robot's UART frames to MQTT:

- **`searchesp32.ino`** – Basic UART→MQTT bridge.
- **`searchesp32_reservation.ino`** – Bridge with intent and goal reservation topics.

A helper script, **`clue_object_generator.py`**, produces random object and clue layouts for testing.

### Algorithm summary

1. **Planning** – Depending on the script, robots either plan full A* paths or select the next cell probabilistically. Planning accounts for turns, center‑ward bias, peer reservations, and clue rewards.
2. **Execution** – Robots turn toward the next cell, move one grid square, and update internal maps. Bumps trigger an immediate stop.
3. **Messaging** – After each action robots send UART frames like `1.x,y;dx,dy` (position), `2.x,y` (visited), `3.x,y` (clue), `4.x,y` (alert/object), `5.x,y` (intent), or `7.x,y` (goal reservation). The ESP32 publishes these to topics named `<clientID><topicDigit>` and relays MQTT commands back over UART.

### Example message flow

1. **Startup** – Robot boots, calibrates sensors, and publishes its starting position with `1.x,y;dx,dy`.
2. **Planning & movement** – Before moving it may reserve the next cell using `5.x,y` and then advances, reporting the move with `2.x,y`.
3. **Exploration updates** – Clues or object hits generate `3.x,y` or `4.x,y` messages.
4. **External commands** – MQTT messages published on `<clientID><topicDigit>` are forwarded to the robot over UART.
5. **Mission end** – On completion or alert, the robot stops and may publish a final position.

## Components

- **`pololu-astar.py`**, **`pololu-astar-reservation.py`**, **`pololu-nextcell.py`**, and **`pololu-sweep.py`** – MicroPython programs for the Pololu 3pi+ 2040 OLED implementing different grid search strategies.
- **`searchesp32.ino`** and **`searchesp32_reservation.ino`** – Arduino sketches that bridge UART messages to an MQTT broker and forward commands back to the robot.
- **`clue_object_generator.py`** – Utility for generating random object and clue locations for trials.

## Hardware setup

1. **Pololu 3pi+ 2040 OLED** running MicroPython.
2. **ESP32** connected to the Pololu via UART (TX=GP28 → RX2, RX=GP29 ← TX2).
3. Wi‑Fi network and MQTT broker accessible to the ESP32.

## Software setup

### Pololu robot

1. Flash MicroPython on the Pololu 3pi+ 2040.
2. Copy one of the robot scripts (`pololu-astar.py`, `pololu-astar-reservation.py`, `pololu-nextcell.py`, or `pololu-sweep.py`) to the device.
3. Adjust settings in the script for your grid size, starting pose, and tuning constants if needed.

### ESP32 bridge

1. Install the Arduino ESP32 core and the `ESP32MQTTClient` library.
2. Open `searchesp32.ino` or `searchesp32_reservation.ino` in the Arduino IDE or PlatformIO.
3. Set your Wi‑Fi SSID, password, MQTT broker URI, and robot `clientID`.
4. Upload the sketch to the ESP32.

## Running

1. Power both the Pololu robot and ESP32.
2. Ensure the ESP32 connects to Wi‑Fi and the MQTT broker.
3. Start the chosen MicroPython script on the robot; it will select and move to new grid cells while exchanging messages with the ESP32 over UART.
4. Monitor MQTT topics such as `<clientID>1` (position), `<clientID>2` (visited), `<clientID>3` (clue), `<clientID>4` (alert/object), `<clientID>5` (intent), and `<clientID>7` (goal reservation when using the reservation variant) to track robot activity.

## Notes

- Modify the UART pins or baud rates in both files if you use different wiring.
- `searchesp32.ino` and `searchesp32_reservation.ino` attempt automatic reconnection to Wi‑Fi and MQTT if the connection drops.
- `pololu-astar.py` and related scripts write debug information to `debug-log.txt` and summary metrics to `metrics-log.txt`.
- Use `clue_object_generator.py` to produce random clue/object layouts for experiments.
- Use the Pololu debug LEDs and serial output for troubleshooting.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

