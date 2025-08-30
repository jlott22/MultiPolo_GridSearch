#include <WiFi.h>
#include "ESP32MQTTClient.h"
#include <vector>

// Wi-Fi credentials
const char *ssid = "USDresearch";
const char *pass = "USDresearch";

// MQTT broker details
const char *server = "mqtt://192.168.1.10:1883"; // MQTT server IP

// Robot-specific identifiers
String clientID = "01";
String pubvisitedtopic = clientID + "2"; //VISITED
String pubpositiontopic = clientID + "1";  //STATUS FOR CURRENT POSITIONING
String pubalerttopic = clientID + "4";  //ALERT FOR OBJECTS FOUND
String pubcluetopic = clientID + "3";  //TOPIC FOR CLUES/FOOTPRINTS/JEWELS FOUND
String pubintenttopic = clientID + "5";  //STATUS FOR CURRENT INTENT
std::vector<String> otherIDs = {"00", "02", "03"};
const char *lastWillMessage = "disconnected"; // Last Will message

ESP32MQTTClient mqttClient; // MQTT client object

// UART Configuration
#define RXD2 16  // UART RX pin
#define TXD2 17  // UART TX pin
HardwareSerial robotSerial(2); // UART2 for Pololu communication

// Reconnection tracking
unsigned long lastReconnectAttempt = 0;
const unsigned long reconnectInterval = 5000; // 5 seconds

void frameToRobot(char topicDigit, const String& senderID, const String& payload);

void onMqttConnect(esp_mqtt_client_handle_t client)
{
    if (mqttClient.isMyTurn(client))
    {
        mqttClient.publish(pubpositiontopic.c_str(), "connected", 0, false);
        Serial.println("Connected to MQTT Broker first time");

        // Subscribe to MQTT topics for each peer
        for (const String &peer : otherIDs)
        {
            for (char topicDigit = '1'; topicDigit <= '5'; ++topicDigit)
            {
                String topic = peer + topicDigit;
                mqttClient.subscribe(topic.c_str(), [topicDigit, peer](const String &payload) {
                    frameToRobot(topicDigit, peer, payload);
                });
            }
        }

        Serial.println("Subscribed to topics first time");
    }
}

void setup()
{
    // Initialize debugging
    Serial.begin(230400);
    robotSerial.begin(230400, SERIAL_8N1, RXD2, TXD2);

    // Connect to Wi-Fi
    Serial.println("Connecting to Wi-Fi...");
    WiFi.begin(ssid, pass);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to Wi-Fi!");

    // MQTT Client Setup
    mqttClient.setURI(server);
    mqttClient.enableDebuggingMessages(); // Enable MQTT debug logs
    mqttClient.enableLastWillMessage(pubpositiontopic.c_str(), lastWillMessage); // Set Last Will message
    mqttClient.setKeepAlive(3); // Keep connection alive with a 5-second timeout

    // Start the MQTT loop
    mqttClient.loopStart();
}

void loop()
{
    // Ensure Wi-Fi is connected
    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("Wi-Fi disconnected. Attempting reconnection...");
        WiFi.begin(ssid, pass);
        while (WiFi.status() != WL_CONNECTED)
        {
            delay(3000);
            Serial.print(".");
        }
        Serial.println("\nWi-Fi reconnected!");
    }

    // Ensure MQTT connection
    if (!mqttClient.isConnected())
    {
        unsigned long currentMillis = millis();
        if (currentMillis - lastReconnectAttempt > reconnectInterval)
        {
            Serial.println("MQTT disconnected. Attempting reconnection...");
            lastReconnectAttempt = currentMillis;

            Serial.println("Reconnected to MQTT broker.");
            mqttClient.publish(pubpositiontopic.c_str(), "Reconnected", 0, false);

            // Resubscribe to topics after reconnection
            for (const String &peer : otherIDs)
            {
                for (char topicDigit = '1'; topicDigit <= '5'; ++topicDigit)
                {
                    String topic = peer + topicDigit;
                    mqttClient.subscribe(topic.c_str(), [topicDigit, peer](const String &payload) {
                        frameToRobot(topicDigit, peer, payload);
                    });
                }
            }

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
          String full_msg = serialBuffer.substring(0, serialBuffer.length());
          serialBuffer = "";
          handlemsg(full_msg); //publish message to proper topic
        }
    }

    delay(1); // Short delay to prevent busy looping
}

void handlemsg(String line) {
  int divider = line.indexOf('.'); //indexes where the message divider (.) is in the string
  if (divider == -1) return;  // dont process ill formed message

  String topic = line.substring(0, divider);
  String message = line.substring(divider + 1);

  // For debug
  Serial.print("tout: "); Serial.println(topic);

  sendtoMQTT(topic, message);
}

void frameToRobot (char topicDigit, const String &senderID, const String &payload) {
  // We keep the '-' end-to-end. If payload already ends with '-', don't add a second one.
  bool hasTerminator = payload.length() && payload.charAt(payload.length()-1) == '-';

  robotSerial.print(senderID);       // "00", "01", ...
  robotSerial.print(topicDigit);     // '1'..'5'
  robotSerial.print('.');
  robotSerial.print(payload);        // very likely already ends with '-'
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
}

void handleMQTT(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    auto *event = static_cast<esp_mqtt_event_handle_t>(event_data);
    mqttClient.onEventCallback(event); // Pass events to the client
}
