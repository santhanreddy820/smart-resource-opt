# src/make_fake_processed.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "processed_sensor_data.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n_devices = 5
rows_per_device = 200

records = []
for d in range(n_devices):
    device_id = f"dev_{d+1}"
    base = float(rng.integers(20, 80))
    ts = pd.date_range("2025-01-01", periods=rows_per_device, freq="H")
    power = base + rng.normal(0, 5, size=rows_per_device).cumsum() * 0.1
    water = 5 + rng.normal(0, 1, size=rows_per_device)
    net = 10 + rng.normal(0, 2, size=rows_per_device)
    # occasional spikes - create synthetic faults
    fault = (rng.random(rows_per_device) < 0.02).astype(int)
    for t, p, w, n_, f in zip(ts, power, water, net, fault):
        records.append({"ts": t.isoformat(), "device_id": device_id, "power_kW": float(p), "water_lpm": float(w), "network_mbps": float(n_), "fault": int(f)})
df = pd.DataFrame.from_records(records)
df.to_csv(OUT, index=False)
print("Wrote synthetic processed file to:", OUT)
