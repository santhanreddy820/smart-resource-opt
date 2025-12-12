import numpy as np
import pandas as pd

def simulate_data(start, end, freq="15min", seed=42):
    np.random.seed(seed)

    idx = pd.date_range(start, end, freq=freq)
    zones = ["North", "South", "East", "West"]

    rows = []
    for zone in zones:
        base_power = {"North": 50, "South": 40, "East": 45, "West": 35}[zone]

        for t in idx:
            hour = t.hour + t.minute/60
            daily_factor = 1 + 0.4*np.sin((hour-6)/24*2*np.pi)
            power = base_power * daily_factor * (1 + np.random.normal(0,0.03))

            water = 20 + 5*np.sin((hour-7)/24*2*np.pi)
            net = 5 + 3*np.sin((hour-9)/24*2*np.pi)

            device_id = f"{zone}_dev_{np.random.randint(1,6)}"

            rows.append([t, zone, device_id, power, water, net, 0])

    df = pd.DataFrame(rows, columns=[
        "ts", "zone", "device_id", "power_kW", "water_lpm", "network_mbps", "fault"
    ])

    return df
