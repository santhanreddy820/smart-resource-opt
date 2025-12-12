# 🚀 Smart Resource Optimization  
### **IoT Fault Detection & Power Forecasting System (FastAPI + ML)**

A production-ready **end-to-end IoT analytics pipeline** that performs real-time **fault detection** and **short-term power forecasting** using Python, Scikit-Learn, and FastAPI.  
The system includes **feature engineering**, **reproducible ML training scripts**, and a fully functional **REST API** for real-time and batch inference.

---

## ⭐ Key Capabilities

### 🔧 Fault Detection (Classification)
- RandomForest-based model for detecting abnormal device behavior  
- Outputs fault probability + recommended action  
- Uses engineered time-based features (lag, rolling average, hour, day-of-week)

### 🔮 Power Forecasting (Regression)
- Predicts next-step power consumption  
- Lightweight RandomForestRegressor  
- Reports MAE / RMSE during training  

### ⚡ FastAPI Service
Real-time ML inference endpoints:
- `POST /predict`  
- `POST /predict_batch`  
- `POST /forecast`  
- `GET /health`  
- Interactive API documentation via **Swagger UI**

### 🧪 Reproducible ML Training
- `train_fault_simple.py`  
- `train_forecast_simple.py`  
- Optional synthetic data generator  

---

## 📁 Project Structure

smart-resource-opt/
│
├── src/
│ ├── api.py
│ ├── train_fault_simple.py
│ ├── train_forecast_simple.py
│ └── make_fake_processed.py
│
├── models/
│ ├── fault_rf.pkl
│ └── forecast_h1.pkl
│
├── data/
│ └── processed/
│ └── sample_processed_sensor_data.csv
│
├── images/
│ ├── 01_api_docs.png
│ ├── 02_health_check.png
│ ├── 03_predict_batch_output.png
│ ├── 04_training_results.png
│ └── 05_project_tree.png
│
├── requirements.txt
├── .gitignore
└── README.md

yaml
Copy code

---

# 🔧 Installation (Windows)

### 1. Clone the repository
```powershell
git clone https://github.com/<your-username>/smart-resource-opt.git
cd smart-resource-opt
2. Create and activate virtual environment
powershell
Copy code
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
3. (Optional) Generate sample data
powershell
Copy code
python .\src\make_fake_processed.py
4. Train models (if not included)
powershell
Copy code
python .\src\train_fault_simple.py
python .\src\train_forecast_simple.py
5. Run the API
powershell
Copy code
uvicorn src.api:app --host 127.0.0.1 --port 8000
Open Swagger documentation:
👉 http://127.0.0.1:8000/docs

📡 API Examples
🩺 Health Check
powershell
Copy code
Invoke-RestMethod http://127.0.0.1:8000/health
🔍 Single Prediction
powershell
Copy code
$body = @{
  ts = (Get-Date).ToString("s")
  device_id = "dev_1"
  power_kW = 60
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict -Method POST -Body $body -ContentType "application/json"
📦 Batch Prediction
powershell
Copy code
$batch = @{
  readings = @(
    @{ ts=(Get-Date).ToString("s"); device_id="dev_1"; power_kW=40 },
    @{ ts=(Get-Date).AddMinutes(5).ToString("s"); device_id="dev_2"; power_kW=70 }
  )
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict_batch -Method POST -Body $batch -ContentType "application/json"
🔮 Forecasting
powershell
Copy code
$body = @{
  ts = (Get-Date).ToString("s")
  device_id = "dev_1"
  power_kW = 55
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/forecast -Method POST -Body $body -ContentType "application/json"
🖼️ Screenshots
📘 API Documentation

🩺 Health Check

📦 Batch Prediction

📉 Training Results

🗂️ Project Structure

🧠 Models
Fault Detection
RandomForestClassifier

Real-time anomaly detection

Features engineered from time-series patterns

Power Forecasting
RandomForestRegressor

Short-term demand prediction model

🛠️ Tech Stack
Layer	Tools
Backend API	FastAPI, Uvicorn
ML Models	Scikit-Learn
Data Processing	Pandas, NumPy
Language	Python
Docs	Swagger UI
OS	Windows

📜 License
MIT License
© 2025 Dasari Santhan Reddy

🎯 Why This Project Stands Out
Demonstrates real-world ML deployment

Combines fault detection + forecasting

Clear training → model → API pipeline

Easy to reproduce and extend

Suitable for ML Engineer, Data Scientist, IoT Analytics roles

🚀 Future Enhancements
Docker containerization

CI/CD (GitHub Actions)

Automated retraining

MQTT/Kafka real-time ingestion pipeline

yaml
Copy code
