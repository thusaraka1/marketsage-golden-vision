# MarketSage - CSE Stock Prediction System

An AI-powered stock market prediction system for the Colombo Stock Exchange (CSE) using BiLSTM + XGBoost ensemble deep learning.

## 📂 Project Structure

```
├── frontend/           # React/Vite web application
├── backend/
│   ├── node_server/    # Express.js (auth, reviews, reports)
│   └── prediction_api/ # Python Flask ML API
├── models/
│   ├── ensemble/       # Trained BiLSTM + XGBoost models
│   └── archive/        # Old/experimental models
├── data/
│   └── historical/     # CSE stock data (2015-2025)
├── training/           # Model training scripts
├── docs/               # Documentation & proposals
└── venv_ml/            # Python virtual environment
```

## 🚀 Quick Start (Manual)

### 1. Frontend (React)
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:8080
```

### 2. Node Backend (Auth & Reviews)
```bash
cd backend/node_server
npm install
npm run dev
# Runs on http://localhost:3001
```

### 3. Prediction API (Python ML)
```bash
cd backend/prediction_api
..\..\..\venv_ml\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

## ⚡ One-Click Deployment (Recommended)

To set up and run on a new machine:

1. **Install Prerequisites**: Ensure you have [Node.js](https://nodejs.org/) and [Python](https://www.python.org/) installed.
2. **Setup**: Double-click `setup.bat` to install all dependencies automatically.
3. **Run**: Double-click `run.bat` to start all servers and open the app.


## 🤖 Model Details

| Model | Accuracy | Features |
|-------|----------|----------|
| BiLSTM + XGBoost Ensemble | ~62% | RSI, MACD, Bollinger Bands, Volatility |

## 📊 Companies Tracked

- **COMB** - Commercial Bank of Ceylon
- **CTC** - Ceylon Tobacco Company
- **DIAL** - Dialog Axiata
- **DIST** - Distilleries Company
- **JKH** - John Keells Holdings

## 📜 License

This project is for research and educational purposes only.
