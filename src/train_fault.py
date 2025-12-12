# src/train_fault.py
"""
Train a fault detection RandomForest classifier and save artifact to models/fault_rf.pkl
Requires: data/processed/processed_sensor_data.csv with at least columns: ts, device_id, power_kW, fault (0/1)
If you do not have a 'fault' column, this script creates a synthetic label based on power threshold.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "processed_sensor_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
OUT = MODELS / "fault_rf.pkl"

if not PROC.exists():
    raise FileNotFoundError(f"Processed CSV not found at {PROC}")

df = pd.read_csv(PROC)
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values(['device_id','ts']).reset_index(drop=True)

# simple feature engineering
df['hour'] = df['ts'].dt.hour
df['dow'] = df['ts'].dt.dayofweek
df['is_weekend'] = (df['dow'] >= 5).astype(int)
df['power_lag_1'] = df.groupby('device_id')['power_kW'].shift(1).fillna(method='bfill')
df['power_roll_1h'] = df.groupby('device_id')['power_kW'].transform(lambda x: x.rolling(4, min_periods=1).mean())

# if no fault column, create a synthetic label for training purposes
if 'fault' not in df.columns:
    # synthetic rule: sudden large jump or high power => label 1
    df['fault'] = ((df['power_kW'] - df['power_lag_1']).abs() > 20).astype(int)

FEATURES = ['power_kW','power_lag_1','power_roll_1h','hour','dow','is_weekend']
df = df.dropna(subset=FEATURES)
X = df[FEATURES]
y = df['fault']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None

print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))
if y_proba is not None and len(np.unique(y_test)) > 1:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(clf, OUT)
print("Saved classifier to:", OUT)
