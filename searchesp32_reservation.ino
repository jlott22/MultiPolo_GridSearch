/*
 * Copyright 2024
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <WiFi.h>
#include "ESP32MQTTClient.h"
#include <vector>
#include <assert.h>

// Wi-Fi network credentials
const char *ssid = "USDresearch";
const char *pass = "USDresearch";

// MQTT broker configuration
const char *server = "mqtt://192.168.1.10:1883"; // MQTT server URI

// Robot‑specific MQTT topics
String clientID = "01";                            // unique robot ID
String pubvisitedtopic = clientID + "2";           // visited cells
String pubpositiontopic = clientID + "1";          // current position
String pubalerttopic = clientID + "4";             // object alerts
String pubcluetopic = clientID + "3";              // clue reports
String pubintenttopic = clientID + "5";            // next intended cell
String pubgoaltopic = clientID + "7";             // goal reservations
const char *hubCommandTopic = "hub/command";       // shared hub command topic
const char *pubresulttopic = "search/results";     // shared search result topic
std::vector<String> otherIDs = {"00", "02", "03"};
const char *lastWillMessage = "disconnected";      // Last Will message

ESP32MQTTClient mqttClient; // MQTT client object

// UART configuration
#define RXD2 16  // UART RX pin
#define TXD2 17  // UART TX pin
HardwareSerial robotSerial(2); // UART2 for communication with the Pololu

// Track reconnection attempts to throttle retries
unsigned long lastReconnectAttempt = 0;
const unsigned long reconnectInterval = 4000; // check every 4 seconds

void frameToRobot(char topicDigit, const String& senderID, const String& payload);

void onMqttConnect(esp_mqtt_client_handle_t client)
{
    if (mqttClient.isMyTurn(client))
    {
        mqttClient.publish(pubpositiontopic.c_str(), "connected", 0, false);

        // Subscribe to MQTT topics for each peer
        for (const String &peer : otherIDs)
        {
            for (char topicDigit = '1'; topicDigit <= '7'; ++topicDigit)
            {
                String topic = peer + topicDigit;
                mqttClient.subscribe(topic.c_str(), [topicDigit, peer](const String &payload) {
                    frameToRobot(topicDigit, peer, payload);
                });
            }
        }

        // Subscribe to hub command topic
        mqttClient.subscribe(hubCommandTopic, [](const String &payload) {
            String hub = "99";
            frameToRobot('6', hub, payload);
        });
    }
}

void setup()
{
    // Initialize serial ports for robot communication
    Serial.begin(115200);
    robotSerial.begin(115200, SERIAL_8N1, RXD2, TXD2);

    // Connect to Wi‑Fi
    WiFi.begin(ssid, pass);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
    }

    // Configure the MQTT client
    mqttClient.setURI(server);
    mqttClient.enableLastWillMessage(pubpositiontopic.c_str(), lastWillMessage); // set Last Will message
    mqttClient.setKeepAlive(3);                                            // 5-second keep-alive timeout

    // Start the MQTT loop
    mqttClient.loopStart();
}

void loop()
{
    // Ensure Wi‑Fi is connected
    if (WiFi.status() != WL_CONNECTED)
    {
        WiFi.begin(ssid, pass);
        while (WiFi.status() != WL_CONNECTED)
        {
            delay(3000);
        }
    }

    // Ensure MQTT connection
    if (!mqttClient.isConnected())
    {
        unsigned long currentMillis = millis();
        if (currentMillis - lastReconnectAttempt > reconnectInterval)
        {
            lastReconnectAttempt = currentMillis;
            mqttClient.publish(pubpositiontopic.c_str(), "Reconnected", 0, false);

            // Resubscribe to topics after reconnection
            for (const String &peer : otherIDs)
            {
                for (char topicDigit = '1'; topicDigit <= '7'; ++topicDigit)
                {
                    String topic = peer + topicDigit;
                    mqttClient.subscribe(topic.c_str(), [topicDigit, peer](const String &payload) {
                        frameToRobot(topicDigit, peer, payload);
                    });
                }
            }

            mqttClient.subscribe(hubCommandTopic, [](const String &payload) {
                String hub = "99";
                frameToRobot('6', hub, payload);
            });

        }
    }

    static String serialBuffer = "";

    // Check for responses from the Pololu robot
    while (robotSerial.available())
    {
      char c = robotSerial.read();
      serialBuffer += c;
        if (c == '-') {
          // Full message received
          serialBuffer.trim(); // remove any unwanted whitespace

          // Remove trailing '-' and process
          String full_msg = serialBuffer.substring(0, serialBuffer.length() - 1);
          serialBuffer = "";
          handlemsg(full_msg); //publish message to proper topic
        }
    }

    delay(1); // Short delay to prevent busy looping
}

void handlemsg(String line) {
  int divider = line.indexOf('.'); // position of the divider in the string
  assert(divider != -1);
  String topic = line.substring(0, divider);
  String message = line.substring(divider + 1);
  sendtoMQTT(topic, message);
}

void frameToRobot (char topicDigit, const String &senderID, const String &payload) {
  // Keep a single '-' terminator end-to-end; avoid adding a second one.
  bool hasTerminator = payload.length() && payload.charAt(payload.length()-1) == '-';

  robotSerial.print(senderID);       // "00", "01", ...
  robotSerial.print(topicDigit);     // '1'..'7'
  robotSerial.print('.');
  robotSerial.print(payload);        // payload typically already ends with '-'
  if (!hasTerminator) robotSerial.print('-');
};


void sendtoMQTT(String topic, String msg) {
  if (topic == "4") {
    mqttClient.publish(pubalerttopic.c_str(), msg, 0, false);
  }
  else if (topic == "2") {
    mqttClient.publish(pubvisitedtopic.c_str(), msg, 0, false);
  }
  else if (topic == "1") {
    mqttClient.publish(pubpositiontopic.c_str(), msg, 0, false);
  }
  else if (topic == "3") {
    mqttClient.publish(pubcluetopic.c_str(), msg, 0, false);
  }
  else if (topic == "5") {
    mqttClient.publish(pubintenttopic.c_str(), msg, 0, false);
  }
  else if (topic == "7") {
    mqttClient.publish(pubgoaltopic.c_str(), msg, 0, false);
  }
  else if (topic == "6") {
    String payload = clientID + ":" + msg;
    mqttClient.publish(pubresulttopic, payload, 0, false);
  }
}

void handleMQTT(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    auto *event = static_cast<esp_mqtt_event_handle_t>(event_data);
    mqttClient.onEventCallback(event); // Pass events to the client
}
