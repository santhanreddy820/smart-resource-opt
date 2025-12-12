Smart Resource Optimization
IoT Fault Detection & Power Forecasting System (FastAPI + Machine Learning)
A production-style IoT analytics system that performs real-time fault detection and short-term power forecasting using Python, Scikit-Learn, and FastAPI.

Features
Fault Detection (Classification)

Detect abnormal device behavior using RandomForest
Outputs fault probability + recommended action

Power Forecasting (Regression)

Predicts next-step power usage
MAE/RMSE performance metrics

FastAPI Microservice

POST /predict
POST /predict_batch
POST /forecast
GET /health
Built-in Swagger UI documentation

ML Training (Reproducible)

train_fault_simple.py
train_forecast_simple.py
Optional synthetic data generator


Project Structure
smart-resource-opt/
│
├── src/
│   ├── api.py
│   ├── train_fault_simple.py
│   ├── train_forecast_simple.py
│   └── make_fake_processed.py
│
├── models/
│   ├── fault_rf.pkl
│   └── forecast_h1.pkl
│
├── data/
│   └── processed/
│       └── sample_processed_sensor_data.csv
│
├── images/
│   ├── 01_api_docs.png
│   ├── 02_health_check.png
│   ├── 03_predict_batch_output.png
│   ├── 04_training_results.png
│   └── 05_project_tree.png
│
├── requirements.txt
├── .gitignore
└── README.md

Installation (Windows)
1. Clone the repository
bashgit clone https://github.com/<your-username>/smart-resource-opt.git
cd smart-resource-opt
2. Create & activate virtual environment
bashpython -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
3. (Optional) Generate sample data
bashpython .\src\make_fake_processed.py
4. Train models
bashpython .\src\train_fault_simple.py
python .\src\train_forecast_simple.py
5. Start the API
bashuvicorn src.api:app --host 127.0.0.1 --port 8000
Open Swagger UI: http://127.0.0.1:8000/docs

API Usage Examples
Health Check
powershellInvoke-RestMethod http://127.0.0.1:8000/health
Single Prediction
powershell$body = @{
    ts = (Get-Date).ToString("s")
    device_id = "dev_1"
    power_kW = 60
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict -Method POST -Body $body -ContentType "application/json"
Batch Prediction
powershell$batch = @{
    readings = @(
        @{ ts=(Get-Date).ToString("s"); device_id="dev_1"; power_kW=40 },
        @{ ts=(Get-Date).AddMinutes(5).ToString("s"); device_id="dev_2"; power_kW=70 }
    )
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/predict_batch -Method POST -Body $batch -ContentType "application/json"
Forecasting
powershell$body = @{
    ts = (Get-Date).ToString("s")
    device_id = "dev_1"
    power_kW = 55
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/forecast -Method POST -Body $body -ContentType "application/json"

Tech Stack
ComponentTechnologyBackend APIFastAPI, UvicornML ModelsScikit-LearnData ProcessingPandas, NumPyLanguagePythonDocsSwagger UIOSWindows


Future Enhancements

Dockerization
CI/CD with GitHub Actions
Kafka/MQTT ingestion
Automated retraining


License
MIT License © 2025 Dasari Santhan Reddy
