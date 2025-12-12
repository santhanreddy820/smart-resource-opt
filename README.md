# Smart Resource Optimization (IoT Fault Detection & Power Forecasting)

A compact end-to-end ML system for IoT analytics, featuring real-time fault detection, power forecasting, and a FastAPI-based inference service. Models are trained using simple reproducible scripts and served through clean REST endpoints.

---

## 🔧 Features
- Fault detection using RandomForestClassifier  
- Short-term power forecasting with RandomForestRegressor  
- REST API endpoints: `/predict`, `/predict_batch`, `/forecast`, `/health`  
- Swagger UI for easy testing  
- Synthetic data generator for quick experimentation  

---

## 📁 Project Structure
smart-resource-opt/
├── src/
│ ├── api.py
│ ├── train_fault_simple.py
│ ├── train_forecast_simple.py
│ └── make_fake_processed.py
├── models/
├── data/processed/
├── images/
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### Install & Setup


python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt


### Train Models


python src/train_fault_simple.py
python src/train_forecast_simple.py


### Run API


uvicorn src.api:app --host 127.0.0.1 --port 8000

Swagger Docs → http://127.0.0.1:8000/docs

---

## 📡 Example Usage

### Health Check


Invoke-RestMethod http://127.0.0.1:8000/health


### Single Prediction


Invoke-RestMethod http://127.0.0.1:8000/predict
 ...


---

## 🛠 Tech Stack
- Python, FastAPI, Scikit-Learn  
- Pandas, NumPy  
- Uvicorn (API server)  

---

## 📄 License
MIT License © 2025 Dasari Santhan Reddy
