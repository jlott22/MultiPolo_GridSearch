#include <WiFi.h>
#include "ESP32MQTTClient.h"

// Wi-Fi credentials
const char *ssid = "USDresearch";
const char *pass = "USDresearch";

// MQTT broker details
const char *server = "mqtt://192.168.1.10:1883"; // MQTT server IP

// Robot-specific identifiers
String clientID = "00";
String statustopic = clientID + "/status";  // Status topic for publishing
// All other topics are handled dynamically from MQTT messages
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

        // Forward any MQTT message published by another robot
        mqttClient.subscribe("+/+",
                            [](const String &topic, const String &payload)
                            {
                                // Ignore messages published by this robot
                                if (topic.startsWith(clientID + "/"))
                                {
                                    return;
                                }

                                Serial.println("Forwarding MQTT message: " + topic + ":" + payload);
                                robotSerial.println(topic + ":" + payload);
                            });

        Serial.println("Subscribed to all robot topics");
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

            // Resubscribe to all robot topics after reconnection
            mqttClient.subscribe("+/+",
                                [](const String &topic, const String &payload)
                                {
                                    if (topic.startsWith(clientID + "/"))
                                    {
                                        return;
                                    }

                                    Serial.println("Forwarding MQTT message: " + topic + ":" + payload);
                                    robotSerial.println(topic + ":" + payload);
                                });
        }
    }

    static String serialBuffer = "";

    // Check for responses from the Pololu robot
    while (robotSerial.available())
    {
        char c = robotSerial.read();
        if (c == '-')
        {
            serialBuffer.trim(); // remove any unwanted whitespace
            handlemsg(serialBuffer); // publish message to proper topic
            serialBuffer = ""; // reset buffer for next message
        }
        else
        {
            serialBuffer += c; // accumulate characters until terminator
        }
    }

    delay(1); // Short delay to prevent busy looping
}

// Parse "<id>/<topic>:<payload>" from Pololu and forward to MQTT
void handlemsg(const String &line)
{
  int slash = line.indexOf('/');
  int colon = line.indexOf(':', slash + 1);
  if (slash == -1 || colon == -1)
  {
    Serial.println("Warning: malformed message from Pololu robot: " + line);
    return;
  }

  String id = line.substring(0, slash);
  String topic = line.substring(slash + 1, colon);
  String message = line.substring(colon + 1);

  // For debug
  Serial.print("ID: "); Serial.println(id);
  Serial.print("Topic: "); Serial.println(topic);
  Serial.print("Message: "); Serial.println(message);

  sendtoMQTT(id, topic, message);
}

// Publish to <id>/<topic> on the MQTT broker
void sendtoMQTT(const String &id, const String &topic, const String &msg)
{
  String fullTopic = id + "/" + topic;
  mqttClient.publish(fullTopic.c_str(), msg, 0, false);
}

void handleMQTT(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    auto *event = static_cast<esp_mqtt_event_handle_t>(event_data);
    mqttClient.onEventCallback(event); // Pass events to the client
}