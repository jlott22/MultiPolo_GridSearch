# Pololu MQTT Search

This repository contains a two-part system for coordinating Pololu 3pi+ 2040 robots through MQTT using an ESP32 bridge.

## Overview

This project coordinates one or more Pololu 3pi+ 2040 robots using an ESP32 as a serial-to-MQTT bridge. The robot runs `pololu-astar.py`, which performs an A* search over a 2-D grid:

1. **Planning** – The script maintains maps of visited cells, clue locations, and probabilities. It plans the next step with A*; costs include turns, center-ward bias for the initial lawn-mower sweep, intent reservations from a peer robot, and accumulated rewards after clues are found.
2. **Execution** – The robot turns toward the next cell, moves one grid square, and updates its internal maps. Bumps trigger an immediate stop.
3. **Messaging** – After each action it sends UART frames like `1.x,y;dx,dy` (position), `2.x,y` (visited), `3.x,y` (clue), `4.x,y` (alert/object), or `5.x,y` (intent). An ESP32 running `searchesp32.ino` listens on UART, publishes to MQTT topics named `<clientID><topicDigit>`, and relays any incoming MQTT commands back to the robot.

### Example message flow

1. **Startup** – Robot boots, calibrates sensors, and publishes its starting position with `1.x,y;dx,dy`.
2. **Planning & movement** – Before moving it may reserve the next cell using `5.x,y` and then advances, reporting the move with `2.x,y`.
3. **Exploration updates** – Clues or object hits generate `3.x,y` or `4.x,y` messages.
4. **External commands** – MQTT messages published on `<clientID><topicDigit>` are forwarded to the robot over UART.
5. **Mission end** – On completion or alert, the robot stops and may publish a final position.

## Components

- **`pololu-astar.py`** – MicroPython program for the Pololu 3pi+ 2040 OLED. It performs a grid search, communicates over UART, and sends position, visited cell, clue, alert, and intent messages to an attached ESP32 that forwards them to MQTT.
- **`searchesp32.ino`** – Arduino sketch for an ESP32 that bridges UART messages from the Pololu robot to an MQTT broker and subscribes to the same topics for commands.

## Hardware setup

1. **Pololu 3pi+ 2040 OLED** running MicroPython.
2. **ESP32** connected to the Pololu via UART (TX=GP28 → RX2, RX=GP29 ← TX2).
3. Wi‑Fi network and MQTT broker accessible to the ESP32.

## Software setup

### Pololu robot

1. Flash MicroPython on the Pololu 3pi+ 2040.
2. Copy `pololu-astar.py` to the device.
3. Adjust settings in the script for your grid size, starting pose, and tuning constants if needed.

### ESP32 bridge

1. Install the Arduino ESP32 core and the `ESP32MQTTClient` library.
2. Open `searchesp32.ino` in the Arduino IDE or PlatformIO.
3. Set your Wi‑Fi SSID, password, MQTT broker URI, and robot `clientID`.
4. Upload the sketch to the ESP32.

## Running

1. Power both the Pololu robot and ESP32.
2. Ensure the ESP32 connects to Wi‑Fi and the MQTT broker.
3. Start the MicroPython script on the robot; it will plan and move using A* search and exchange messages with the ESP32 over UART.
4. Monitor MQTT topics such as `<clientID>1` (position), `<clientID>2` (visited), `<clientID>3` (clue), `<clientID>4` (alert/object), and `<clientID>5` (intent) to track robot activity.

## Notes

- Modify the UART pins or baud rates in both files if you use different wiring.
- `searchesp32.ino` attempts automatic reconnection to Wi‑Fi and MQTT if the connection drops.
- `pololu-astar.py` writes debug information to `debug-log.txt` and summary metrics to `metrics-log.txt`.
- Use the Pololu debug LEDs and serial output for troubleshooting.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

