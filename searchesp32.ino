#include <WiFi.h>
#include "ESP32MQTTClient.h"

// Wi-Fi credentials
const char *ssid = "USDresearch";
const char *pass = "USDresearch";

// MQTT broker details
const char *server = "mqtt://192.168.1.10:1883"; // MQTT server IP

// Robot-specific identifiers
String clientID = "00";
String commmandtopic = clientID + "/command"; // Robot-specific command topic
String statustopic = clientID + "/status";  // Status topic for publishing
String alerttopic = clientID + "/alert";  // Alert topic for publishing
String coordtopic = clientID + "/coord";  // Coordinates topic for publishing
const char *lastWillMessage = "disconnected"; // Last Will message

ESP32MQTTClient mqttClient; // MQTT client object

// UART Configuration
#define RXD2 16  // UART RX pin
#define TXD2 17  // UART TX pin
HardwareSerial robotSerial(2); // UART2 for Pololu communication

// Reconnection tracking
unsigned long lastReconnectAttempt = 0;
const unsigned long reconnectInterval = 5000; // 5 seconds

void onMqttConnect(esp_mqtt_client_handle_t client)
{
    if (mqttClient.isMyTurn(client))
    {
        mqttClient.publish(statustopic.c_str(), "connected", 0, false);
        Serial.println("Connected to MQTT Broker!");

        mqttClient.subscribe(commmandtopic.c_str(), [](const String &payload)
                             {
                                 Serial.println("Robot-specific command: " + payload);
                                 robotSerial.println(payload);
                             });

        Serial.println("Subscribed to robot specific command topic");
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
    mqttClient.enableLastWillMessage(statustopic.c_str(), lastWillMessage); // Set Last Will message
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
            delay(1000);
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
            mqttClient.publish(statustopic.c_str(), "Reconnected", 0, false);

            // Resubscribe to topics after reconnection
            mqttClient.subscribe(commmandtopic.c_str(), [](const String &payload)
                                 {
                                     Serial.println("Robot-specific command: " + payload);
                                     robotSerial.println(payload);
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
        else
        {
            Serial.println("Warning: Empty message from Pololu robot");
        }
    }

    delay(1); // Short delay to prevent busy looping
}

void handlemsg(String line) {
  int divider = line.indexOf('='); //indexes where the message divider (=) is in the string
  if (divider == -1) return;  // dont process ill formed message

  String topic = line.substring(0, divider);
  String message = line.substring(divider + 1);

  // For debug
  Serial.print("Topic: "); Serial.println(topic);
  Serial.print("Message: "); Serial.println(message);

  sendtoMQTT(topic, message);
}

void sendtoMQTT(String topic, String msg) {
  if (topic == "alert") {
    mqttClient.publish(alerttopic.c_str(), msg, 0, false);
  }
  else if (topic == "coord") {
    mqttClient.publish(coordtopic.c_str(), msg, 0, false);
  }
  else if (topic == "status") {
    mqttClient.publish(statustopic.c_str(), msg, 0, false);
  }
}

void handleMQTT(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    auto *event = static_cast<esp_mqtt_event_handle_t>(event_data);
    mqttClient.onEventCallback(event); // Pass events to the client
}