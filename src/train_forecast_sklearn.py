# src/train_forecast_sklearn.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "processed_sensor_data.csv"
OUT_CSV = ROOT / "data" / "processed" / "predictions_h1.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(PROC)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values(["device_id","ts"]).reset_index(drop=True)
df["target"] = df.groupby("device_id")["power_kW"].shift(-1)
df = df.dropna(subset=["target"])

df["hour"] = df["ts"].dt.hour
df["dow"] = df["ts"].dt.dayofweek
df["is_weekend"] = (df["dow"] >= 5).astype(int)

X = df[["power_kW","hour","dow","is_weekend"]]
y = df["target"]

split = int(0.8 * len(df))
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(Xtr, ytr)
preds = model.predict(Xte)

print("MAE:", mean_absolute_error(yte, preds))
print("RMSE:", mean_squared_error(yte, preds, squared=False))

joblib.dump(model, MODELS / "forecast_h1.pkl")
print("Saved model → models/forecast_h1.pkl")

df["predicted_power"] = model.predict(X)
df[["ts","device_id","power_kW","predicted_power"]].to_csv(OUT_CSV, index=False)
print("Saved predictions → data/processed/predictions_h1.csv")
