# src/train_forecast.py
"""Train a next-step forecasting model and save predictions CSV for inspection.
Assumes processed CSV exists at data/processed/processed_sensor_data.csv
Saves model to models/forecast_h1.pkl and predictions to data/processed/predictions_h1.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]  # project root
PROC_CSV = ROOT / "data" / "processed" / "processed_sensor_data.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "data" / "processed" / "predictions_h1.csv"

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values(['device_id', 'ts']).reset_index(drop=True)
    # ensure basic time features exist
    df['hour'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    # if rolling/lag features missing, create simple ones
    if 'power_roll_1h' not in df.columns:
        df['power_roll_1h'] = df.groupby('device_id')['power_kW'].transform(lambda x: x.rolling(4, min_periods=1).mean())
    if 'power_roll_6h' not in df.columns:
        df['power_roll_6h'] = df.groupby('device_id')['power_kW'].transform(lambda x: x.rolling(24, min_periods=1).mean())
    if 'power_lag_1' not in df.columns:
        df['power_lag_1'] = df.groupby('device_id')['power_kW'].shift(1).fillna(method='bfill')
    return df

def train_save(horizon=1):
    print("Loading processed data from:", PROC_CSV)
    df = pd.read_csv(PROC_CSV)
    df = prepare_df(df)
    # create target as next-step power_kW (horizon steps ahead)
    target_col = f"target_h{horizon}"
    df[target_col] = df.groupby('device_id')['power_kW'].shift(-horizon)
    # drop rows with no target (end of sequence)
    df_train = df.dropna(subset=[target_col]).reset_index(drop=True)
    # select features
    FEATURES = ['power_kW','power_roll_1h','power_roll_6h','power_lag_1','hour','dow','is_weekend']
    X = df_train[FEATURES]
    y = df_train[target_col]
    # simple temporal split
    split_idx = int(0.8 * len(df_train))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    # LightGBM params (simple)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval], early_stopping_rounds=50, num_boost_round=1000, verbose_eval=100)
    # evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Forecast MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    # save model
    model_path = MODELS_DIR / f"forecast_h{horizon}.pkl"
    joblib.dump(model, model_path)
    print("Saved forecast model to:", model_path)
    # prepare predictions dataset for inspection
    preds_all = model.predict(df_train[FEATURES])
    df_out = df_train[['ts','device_id','power_kW']].copy()
    df_out['predicted_power'] = preds_all
    df_out.to_csv(OUT_CSV, index=False)
    print("Saved predictions CSV to:", OUT_CSV)
    return model

if __name__ == "__main__":
    train_save(horizon=1)
