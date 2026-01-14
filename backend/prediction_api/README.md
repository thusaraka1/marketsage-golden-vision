# Prediction API

Flask-based REST API serving real-time stock predictions using the trained BiLSTM + XGBoost ensemble model.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/companies` | List available companies |
| GET | `/api/predict/<SYMBOL>` | Get prediction for a stock |

## Running

```bash
# Activate virtual environment
..\..\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
python app.py
```

Server runs on `http://localhost:5000`

## Response Example

```json
{
  "symbol": "COMB",
  "signal": "BULLISH",
  "confidence": 72,
  "target": 205.50,
  "currentPrice": 201.25,
  "probabilities": {
    "bilstm": 0.65,
    "xgboost": 0.58,
    "ensemble": 0.61
  }
}
```
