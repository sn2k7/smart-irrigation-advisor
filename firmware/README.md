# ESP32 reference node

`esp32_soil_node/esp32_soil_node.ino` shows how a low-cost node talks to the backend:

1. Register the node once: `POST /api/v1/devices` with `{"device_id": "esp32-node-01", "field_id": 1}`.
2. Every 15 minutes the node posts the compact payload to `POST /api/v1/iot/ingest?recommend=true`:
   `{"device_id": "...", "sm": 19.3, "t": 28.4, "h": 61, "r": 0.0, "bat": 3.9, "rssi": -70}`.
3. The response contains `cmd.irrigate` and `cmd.duration_min`; the node opens the relay for that time
   (with a local 4-hour safety cap and valve closed on boot).

Hardware (approx. cost INR 1,500-2,500): ESP32 dev board, capacitive soil-moisture probe v1.2/v2.0,
DHT22, tipping-bucket rain gauge (0.2 mm/tip), 5 V relay + solenoid valve, 18650 cell + solar charger.

**Calibrate each soil probe** (raw ADC in air and in saturated soil) - uncalibrated capacitive probes can
be off by several % VWC, which directly shifts the computed root-zone depletion.

The firmware is a reference implementation and has not been validated on hardware in this project.
Without hardware, run `python scripts/iot_device_simulator.py --field-id 1`, which sends the same payloads.
