# src/train_forecast_simple.py  (fixed for older sklearn)
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "processed_sensor_data.csv"
MODELS = ROOT / "models"
OUT = ROOT / "data" / "processed" / "predictions_h1.csv"
MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC)
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values(['device_id','ts']).reset_index(drop=True)

# 1-step ahead target
df['target'] = df.groupby('device_id')['power_kW'].shift(-1)
df = df.dropna(subset=['target'])

# features
df['hour'] = df['ts'].dt.hour
df['dow'] = df['ts'].dt.dayofweek
df['power_lag_1'] = df.groupby('device_id')['power_kW'].shift(1).bfill()
df['power_roll_1h'] = df.groupby('device_id')['power_kW'].transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)

FEATURES = ['power_kW','power_lag_1','power_roll_1h','hour','dow']
X = df[FEATURES]
y = df['target']

# split
split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# MAE
mae = mean_absolute_error(y_test, preds)
print("MAE:", mae)

# RMSE (manual sqrt, compatible with all sklearn versions)
rmse = mean_squared_error(y_test, preds) ** 0.5
print("RMSE:", rmse)

# save model
joblib.dump(model, MODELS / "forecast_h1.pkl")

# predictions CSV
df['predicted_power'] = model.predict(X)
df[['ts','device_id','power_kW','predicted_power']].to_csv(OUT, index=False)

print("Saved forecast model and predictions to:", OUT)
