# Pololu MQTT Search

This repository contains a two-part system for coordinating Pololu 3pi+ 2040 robots through MQTT using an ESP32 bridge.

- **`pololu-astar.py`** – MicroPython program for the Pololu 3pi+ 2040 OLED. It performs a grid search, communicates over UART, and sends status, visited cells, clues, and alerts to an attached ESP32 that forwards them to MQTT.
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
4. Monitor MQTT topics such as `<clientID>/status`, `<clientID>/visited`, `<clientID>/clue`, and `<clientID>/alert` to track robot activity.

## Notes

- Modify the UART pins or baud rates in both files if you use different wiring.
- `searchesp32.ino` attempts automatic reconnection to Wi‑Fi and MQTT if the connection drops.
- Use the Pololu debug LEDs and serial output for troubleshooting.

## License

This project is provided as-is without a specific license. Add a license file if you intend to distribute or reuse the code.
