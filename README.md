🚀 Smart Resource Optimization
IoT Fault Detection & Power Forecasting System (FastAPI + ML)

A production-ready end-to-end IoT analytics pipeline that performs real-time fault detection and short-term power forecasting using Python, Scikit-Learn, and FastAPI.
The system includes feature engineering, reproducible ML training scripts, and a fully functional REST API for real-time and batch inference.

⭐ Key Capabilities
🔧 Fault Detection (Classification)

RandomForest-based model for detecting abnormal device behavior

Outputs fault probability + recommended action

Powered by engineered features: lags, rolling averages, hour, weekday

🔮 Power Forecasting (Regression)

Predicts next-step power consumption

Lightweight RandomForestRegressor implementation

Reports MAE/RMSE during training

⚡ FastAPI Service

Provides real-time and batch ML inference via:

POST /predict

POST /predict_batch

POST /forecast

GET /health

Interactive docs → Swagger UI

🧪 Reproducible ML Training

train_fault_simple.py

train_forecast_simple.py

Includes optional synthetic data generator for demos

📁 Project Structure
smart-resource-opt/
│
├── src/
│   ├── api.py                    # FastAPI app
│   ├── train_fault_simple.py     # Fault model training
│   ├── train_forecast_simple.py  # Forecast model training
│   └── make_fake_processed.py    # Sample data generator
│
├── models/
│   ├── fault_rf.pkl
│   └── forecast_h1.pkl
│
├── data/
│   └── processed/
│       └── sample_processed_sensor_data.csv
│
├── images/                       # Screenshots (docs, outputs, UI)
├── requirements.txt
├── .gitignore
└── README.md

🔧 Installation (Windows)
1. Clone the repository
git clone https://github.com/<your-username>/smart-resource-opt.git
cd smart-resource-opt

2. Set up environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

3. (Optional) Generate sample data
python .\src\make_fake_processed.py

4. Train models (if not using provided ones)
python .\src\train_fault_simple.py
python .\src\train_forecast_simple.py

5. Run the API
uvicorn src.api:app --host 127.0.0.1 --port 8000


Open Swagger docs:
👉 http://127.0.0.1:8000/docs

📡 API Examples
🩺 Health Check
Invoke-RestMethod http://127.0.0.1:8000/health

🔍 Single Prediction
$body = @{
  ts = (Get-Date).ToString("s")
  device_id = "dev_1"
  power_kW = 60
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict -Method POST -Body $body -ContentType "application/json"

📦 Batch Prediction
$batch = @{
  readings = @(
    @{ ts=(Get-Date).ToString("s"); device_id="dev_1"; power_kW=40 },
    @{ ts=(Get-Date).AddMinutes(5).ToString("s"); device_id="dev_2"; power_kW=70 }
  )
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict_batch -Method POST -Body $batch -ContentType "application/json"

🔮 Forecasting
$body = @{
  ts = (Get-Date).ToString("s")
  device_id = "dev_1"
  power_kW = 55
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/forecast -Method POST -Body $body -ContentType "application/json"

🖼️ Screenshots

Place the following images under images/ and reference them:

📘 API Documentation

🩺 Health Check

📦 Batch Prediction Output

📉 Training Results

🗂️ Project Structure

🧠 Models
Fault Detection

RandomForestClassifier

Designed for IoT anomaly detection

Trained on engineered time-based features

Power Forecasting

RandomForestRegressor

Predicts short-term power demand

Lightweight and deployment-friendly

🛠️ Tech Stack
Layer	Technology
Backend API	FastAPI, Uvicorn
ML Models	Scikit-Learn
Data Processing	Pandas, NumPy
Language	Python
Environment	venv
Documentation	Swagger UI
OS	Windows
📜 License

MIT License
© 2025 Dasari Santhan Reddy
