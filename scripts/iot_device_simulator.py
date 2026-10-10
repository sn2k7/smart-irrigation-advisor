"""Simulate an ESP32 soil-moisture node posting to a RUNNING API (standard library only).

Start the API first (uvicorn), then in another PowerShell window:
    python scripts/iot_device_simulator.py --field-id 1 --messages 8
    python scripts/iot_device_simulator.py --field-id 1 --inject-faults --api-key changeme

Each message uses the compact payload of POST /api/v1/iot/ingest and asks the
server for a valve command (?recommend=true), exactly like the firmware in
firmware/esp32_soil_node/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta


def post(url: str, payload: dict, api_key: str | None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Device-Key"] = api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--field-id", type=int, required=True)
    parser.add_argument("--device-id", default="esp32-sim-01")
    parser.add_argument("--messages", type=int, default=8)
    parser.add_argument("--interval-s", type=float, default=0.5, help="real seconds between messages")
    parser.add_argument("--step-min", type=int, default=15, help="simulated minutes between readings")
    parser.add_argument("--start-moisture", type=float, default=22.0)
    parser.add_argument("--inject-faults", action="store_true", help="send a spike and a dropout")
    parser.add_argument("--api-key", default=None, help="value for X-Device-Key if SIA_IOT_API_KEY is set")
    args = parser.parse_args()

    status, body = post(f"{args.base_url}/api/v1/devices",
                        {"device_id": args.device_id, "field_id": args.field_id, "description": "IoT simulator"}, None)
    print(f"register device -> {status} {body}")
    if status >= 400:
        raise SystemExit("Device registration failed - is the API running and does the field exist?")

    rng = random.Random(7)
    start = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=args.step_min * (args.messages - 1))
    moisture = args.start_moisture
    for i in range(args.messages):
        moisture -= rng.uniform(0.05, 0.25)  # drying soil
        sm: float | None = round(moisture + rng.gauss(0, 0.3), 1)
        if args.inject_faults and i == args.messages // 2:
            sm = round(moisture + 25, 1)  # spike
        if args.inject_faults and i == args.messages - 2:
            sm = None  # dropout
        ts = start + timedelta(minutes=args.step_min * i)
        payload = {
            "device_id": args.device_id, "sm": sm, "t": round(30 + rng.gauss(0, 0.5), 1),
            "h": round(55 + rng.gauss(0, 2), 1), "r": 0.0, "bat": round(4.1 - 0.01 * i, 2), "rssi": -67,
            "ts": ts.isoformat(timespec="seconds"),
        }
        status, body = post(f"{args.base_url}/api/v1/iot/ingest?recommend=true", payload, args.api_key)
        cmd = body.get("cmd") or {}
        print(f"[{ts:%H:%M}] sm={sm} -> {status} valid={body.get('valid')} issues={body.get('issues')} "
              f"cmd: irrigate={cmd.get('irrigate')} {cmd.get('duration_min')} min, next check {cmd.get('next_check_min')} min")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
