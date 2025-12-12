# src/train_fault_simple.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "processed_sensor_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
OUT = MODELS / "fault_rf.pkl"

df = pd.read_csv(PROC)
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values(['device_id','ts']).reset_index(drop=True)

df['hour'] = df['ts'].dt.hour
df['dow'] = df['ts'].dt.dayofweek
df['power_lag_1'] = df.groupby('device_id')['power_kW'].shift(1).fillna(method='bfill')
df['power_roll_1h'] = df.groupby('device_id')['power_kW'].transform(lambda x: x.rolling(4, min_periods=1).mean())

FEATURES = ['power_kW','power_lag_1','power_roll_1h','hour','dow']
df = df.dropna(subset=FEATURES)
X = df[FEATURES]
y = df['fault'] if 'fault' in df.columns else ((df['power_kW'] - df['power_lag_1']).abs() > 20).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Classification report:")
print(classification_report(y_test, clf.predict(X_test), zero_division=0))

joblib.dump(clf, OUT)
print("Saved classifier to:", OUT)
