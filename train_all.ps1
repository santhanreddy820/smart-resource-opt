# train_all.ps1 (stable-safe version)
$ErrorActionPreference = "Stop"
Write-Host "START: Smart Resource Optimization Pipeline"

# Resolve root directory
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrWhiteSpace($ROOT)) {
    $ROOT = (Get-Location).Path
}
Write-Host ("Project root: " + $ROOT)

# Paths
$venvPath = Join-Path $ROOT "venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
$reqFile = Join-Path $ROOT "requirements.txt"
$modelsDir = Join-Path $ROOT "models"
$dataProcessedDir = Join-Path $ROOT "data\processed"
$trainScript = Join-Path $ROOT "src\train_forecast.py"
$fallbackPath = Join-Path $ROOT "src\train_forecast_sklearn.py"

# 1) Create and activate venv
if (-Not (Test-Path $venvPath)) {
    Write-Host ("Creating venv at " + $venvPath)
    python -m venv $venvPath
}

if (Test-Path $activateScript) {
    Write-Host "Activating venv"
    . $activateScript
} else {
    Write-Host "Warning: Cannot find activate script"
}

# 2) Install dependencies
if (Test-Path $reqFile) {
    Write-Host "Installing dependencies"
    try {
        pip install --upgrade pip
        pip install -r $reqFile
    } catch {
        Write-Host "Failed to install from requirements. Installing minimal set."
        pip install fastapi uvicorn pandas numpy scikit-learn joblib pydantic python-dateutil pytest requests
    }
} else {
    Write-Host "requirements.txt not found. Installing minimal set."
    pip install fastapi uvicorn pandas numpy scikit-learn joblib pydantic python-dateutil pytest requests
}

# 3) Ensure folders exist
if (-Not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}
if (-Not (Test-Path $dataProcessedDir)) {
    New-Item -ItemType Directory -Path $dataProcessedDir | Out-Null
}

# 4) Run LightGBM trainer
$trainSuccess = $false

if (Test-Path $trainScript) {
    Write-Host "Running LightGBM forecast trainer"
    try {
        python $trainScript
        $trainSuccess = $true
    } catch {
        Write-Host "LightGBM failed. Will switch to sklearn fallback."
    }
} else {
    Write-Host "train_forecast.py not found"
    exit 1
}

# 5) Sklearn fallback if needed
if (-Not $trainSuccess) {
    Write-Host ("Writing fallback trainer to " + $fallbackPath)

    $fallbackCode = @"
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "processed_sensor_data.csv"
OUT = ROOT / "data" / "processed" / "predictions_h1.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values(["device_id", "ts"]).reset_index(drop=True)
df["target"] = df.groupby("device_id")["power_kW"].shift(-1)
df = df.dropna(subset=["target"])

df["hour"] = df["ts"].dt.hour
df["dow"] = df["ts"].dt.dayofweek
df["weekend"] = (df["dow"] >= 5).astype(int)

X = df[["power_kW", "hour", "dow", "weekend"]]
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
df["predicted_power"] = model.predict(X)
df.to_csv(OUT, index=False)
"@

    Set-Content -Path $fallbackPath -Value $fallbackCode -Encoding UTF8

    Write-Host "Running sklearn fallback trainer"
    python $fallbackPath
}

# 6) Start FastAPI server
Write-Host "Starting FastAPI on port 8000"
Start-Process -FilePath "python" -ArgumentList "-m uvicorn src.api_app:app --reload --host 0.0.0.0 --port 8000" -WorkingDirectory $ROOT

Write-Host "DONE"
Write-Host "Open browser: http://127.0.0.1:8000/docs"
