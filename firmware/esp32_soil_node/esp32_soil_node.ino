/*
 * Smart Irrigation Advisor - ESP32 reference sensor/valve node
 * -----------------------------------------------------------
 * Reads a capacitive soil-moisture probe, a DHT22 and a tipping-bucket rain
 * gauge, posts a compact JSON message to the backend and switches a valve
 * relay according to the returned command.
 *
 * REFERENCE FIRMWARE: it compiles in the Arduino IDE (ESP32 core 2.x/3.x,
 * libraries "DHT sensor library" + "ArduinoJson" v7) but was not validated on
 * physical hardware as part of this project. The backend can be demonstrated
 * without hardware using scripts/iot_device_simulator.py.
 *
 * Wiring (change the pins below to match your board):
 *   Capacitive soil probe AOUT -> GPIO34 (ADC1)   DHT22 DATA -> GPIO4
 *   Rain gauge reed switch      -> GPIO27 (to GND) Valve relay IN -> GPIO26
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// ---------------- configuration ----------------
const char* WIFI_SSID     = "YOUR_WIFI";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
const char* API_URL       = "http://192.168.1.50:8000/api/v1/iot/ingest?recommend=true";
const char* DEVICE_ID     = "esp32-node-01";   // register with POST /api/v1/devices
const char* DEVICE_KEY    = "";                // value of SIA_IOT_API_KEY (empty = disabled)

const int PIN_SOIL = 34;
const int PIN_DHT = 4;
const int PIN_RAIN = 27;
const int PIN_VALVE = 26;

// Two-point calibration of the capacitive probe: raw ADC in dry air / in
// saturated soil, mapped to volumetric water content (%). Calibrate per probe!
const int SOIL_RAW_DRY = 3300;
const int SOIL_RAW_WET = 1350;
const float SOIL_VWC_DRY = 3.0;
const float SOIL_VWC_WET = 45.0;

const float RAIN_MM_PER_TIP = 0.2;
const uint32_t REPORT_INTERVAL_MS = 15UL * 60UL * 1000UL;  // 15 min
const uint32_t MAX_VALVE_MS = 240UL * 60UL * 1000UL;       // local safety cap: 4 h

DHT dht(PIN_DHT, DHT22);
volatile uint32_t rainTips = 0;
volatile uint32_t lastTipMs = 0;
uint32_t lastReportMs = 0;
uint32_t valveOffAtMs = 0;
bool valveOpen = false;

void IRAM_ATTR onRainTip() {
  uint32_t now = millis();
  if (now - lastTipMs > 200) {  // debounce reed switch
    rainTips++;
    lastTipMs = now;
  }
}

float readSoilVwc() {
  long sum = 0;
  for (int i = 0; i < 16; i++) {  // average to reduce ADC noise
    sum += analogRead(PIN_SOIL);
    delay(5);
  }
  float raw = sum / 16.0;
  float vwc = SOIL_VWC_DRY + (raw - SOIL_RAW_DRY) * (SOIL_VWC_WET - SOIL_VWC_DRY) / (SOIL_RAW_WET - SOIL_RAW_DRY);
  return vwc;  // out-of-range values are sent as-is: the server flags them
}

float readBatteryVolts() {
  return 3.9;  // replace with a voltage-divider reading on your board
}

void setValve(bool open, uint32_t durationMs) {
  if (open) {
    durationMs = min(durationMs, MAX_VALVE_MS);
    valveOffAtMs = millis() + durationMs;
  }
  valveOpen = open;
  digitalWrite(PIN_VALVE, open ? HIGH : LOW);
  Serial.printf("Valve %s (%lu min)\n", open ? "OPEN" : "CLOSED", (unsigned long)(durationMs / 60000UL));
}

void connectWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
}

void report() {
  connectWifi();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi unavailable - keeping valve state, retrying next cycle");
    return;
  }
  noInterrupts();
  float rainMm = rainTips * RAIN_MM_PER_TIP;
  rainTips = 0;
  interrupts();

  JsonDocument doc;
  doc["device_id"] = DEVICE_ID;
  doc["sm"] = readSoilVwc();
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t)) doc["t"] = t;   // missing values are simply omitted
  if (!isnan(h)) doc["h"] = h;
  doc["r"] = rainMm;
  doc["bat"] = readBatteryVolts();
  doc["rssi"] = WiFi.RSSI();
  // No "ts": the server stamps the message (no RTC needed on the node).

  String body;
  serializeJson(doc, body);
  HTTPClient http;
  http.begin(API_URL);
  http.addHeader("Content-Type", "application/json");
  if (strlen(DEVICE_KEY) > 0) http.addHeader("X-Device-Key", DEVICE_KEY);
  int code = http.POST(body);
  if (code == 200) {
    JsonDocument resp;
    if (!deserializeJson(resp, http.getString())) {
      JsonObject cmd = resp["cmd"];
      if (!cmd.isNull()) {
        bool irrigate = cmd["irrigate"] | false;
        float minutes = cmd["duration_min"] | 0.0;
        if (irrigate && minutes > 0 && !valveOpen) setValve(true, (uint32_t)(minutes * 60000.0));
      }
    }
  } else {
    Serial.printf("POST failed: %d\n", code);
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_VALVE, OUTPUT);
  digitalWrite(PIN_VALVE, LOW);  // fail-safe: valve closed on boot
  pinMode(PIN_RAIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_RAIN), onRainTip, FALLING);
  analogReadResolution(12);
  dht.begin();
  connectWifi();
  report();
  lastReportMs = millis();
}

void loop() {
  if (valveOpen && (int32_t)(millis() - valveOffAtMs) >= 0) setValve(false, 0);
  if (millis() - lastReportMs >= REPORT_INTERVAL_MS) {
    report();
    lastReportMs = millis();
  }
  delay(200);
}
