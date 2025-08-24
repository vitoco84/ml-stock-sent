# 📈 Stock Market Forecast with FinBERT Sentiment Analysis

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=vitoco84_ml-stock-sent&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=vitoco84_ml-stock-sent)

Forecast stock prices by combining:
- 📊 Historical market data (Yahoo Finance)
- 📰 News sentiment via [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)
- 🤖 Optional headline generation with local LLMs (Ollama)
- ⚡ A production-ready FastAPI backend + Streamlit dashboard

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10+**
- (Optional) NVIDIA GPU with CUDA for accelerated FinBERT

### Setup
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
# or, with Poetry:
poetry install
```

### Verify CUDA (optional)
```bash
nvidia-smi          # confirm driver & CUDA runtime
# Example: install PyTorch for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📂 Project Structure

- `src/` – data processing, feature engineering, model training
- `app/api/` – FastAPI application
- `app/ui/` – Streamlit dashboard
- `notebooks/` – Jupyter notebooks for exploration & training

---

## ⚡ FastAPI (Backend)

Start the API:
```bash
uvicorn app.api.main:app --reload
```
- Swagger docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/healthz>

### Environment Variables
Create `.env` (do not commit). See [.env.example](.env.example) for all keys.
```
NEWS_API_KEY=your_newsapi_key
OLLAMA_URL=http://localhost:11434      # optional
OLLAMA_MODEL=llama3                    # optional
API_ROOT_PATH=/
CORS_ORIGINS=["http://localhost:8501","http://localhost:3000"]
```

> Note: your code checks a live endpoint when `enrich=true`. Ensure `OLLAMA_URL` is reachable.

---

## 📡 API Endpoints & Schemas

### `GET /healthz`
**Response**
```json
{ "ok": true }
```

---

### `GET /price-history`
Fetch historical stock data.

**Query params**
- `symbol` *(str, required)* — e.g. `AAPL`, `^DJI`
- `end_date` *(YYYY-MM-DD, required)*
- `days` *(int, default 90)* — calendar days to look back

**Response — `PriceHistoryResponse`**
```json
{
  "price": [
    {
      "date": "2025-06-01",
      "open": 180.0,
      "high": 182.5,
      "low": 178.2,
      "close": 181.7,
      "adj_close": 181.7,
      "volume": 10000000
    }
  ]
}
```

**Error cases**
- `404` if no data returned for the symbol/date range
- `500` on unexpected failure

---

### `GET /news-history`
Fetch recent news headlines via NewsAPI.

**Query params**
- `query` *(str, required)* — search term, e.g. `Apple`
- `end_date` *(YYYY-MM-DD, required)*
- `days` *(int, default 7)* — lookback window

**Success — `NewsHistoryResponse`**
```json
{
  "news": [
    { "date": "2025-01-10", "headline": "Apple stock surges" },
    { "date": "2025-01-11", "headline": "Tech sector rallies" }
  ]
}
```
**If no results**
```json
{
  "news": [],
  "message": "No news found."
}
```
**Error cases**
- `500` with message `"Missing NEWS_API_KEY environment variable"` if the key is not configured
- `500` on unexpected failure

---

### `POST /predict-raw`
Predict next prices using historical **price** data and optional **news**.

**Query params**
- `symbol` *(str, required)* — Ticker symbol for context (e.g., `AAPL`)
- `enrich` *(bool, default `false`)* — generate missing headlines locally (requires reachable `OLLAMA_URL`)
- `horizon` *(int, default `30`, min `1`)* — number of steps to return
- `return_path` *(bool, default `true`)* — whether to return full H‑step paths

**Request body — `PredictionRequest`**
```json
{
  "price": [
    { "date": "2025-01-08", "open": 100, "high": 105, "low": 99, "close": 104, "adj_close": 104, "volume": 1000000 },
    { "date": "2025-01-09", "open": 104, "high": 106, "low": 103, "close": 105, "adj_close": 105, "volume": 1200000 }
  ],
  "news": [
    { "date": "2025-01-08", "headline": "Apple launches new product" },
    { "date": "2025-01-09", "headline": "Market opens higher" }
  ]
}
```

**Response — `PredictionResponse` (when `return_path=true`)**
```json
{
  "horizon": 5,
  "log_return": 0.0035,
  "current_price": 105.0,
  "predicted_price": 105.37,
  "log_return_path": [0.0035, 0.0012, 0.0007, -0.0003, 0.0021],
  "predicted_price_path": [105.37, 105.50, 105.57, 105.54, 105.77],
  "predicted_dates": ["2025-01-10", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16"],
  "last_date": "2025-01-09"
}
```
**Response — `PredictionResponse` (when `return_path=false`)**
```json
{
  "horizon": 5,
  "log_return": 0.0035,
  "current_price": 105.0,
  "predicted_price": 105.37
}
```

**Validation & errors**
- `422` if `price` is missing/empty
- `500` on feature generation or model errors

**cURL example**
```bash
curl -X POST "http://localhost:8000/predict-raw?symbol=AAPL&horizon=5&return_path=true&enrich=false"   -H "Content-Type: application/json"   -d '{
        "price":[
          {"date":"2025-01-08","open":100,"high":105,"low":99,"close":104,"adj_close":104,"volume":1000000},
          {"date":"2025-01-09","open":104,"high":106,"low":103,"close":105,"adj_close":105,"volume":1200000}
        ],
        "news":[
          {"date":"2025-01-08","headline":"Apple launches new product"},
          {"date":"2025-01-09","headline":"Market opens higher"}
        ]
      }'
```

---

## 📓 Notebooks (Pipeline)

1. **01_eda.ipynb** – Exploratory Data Analysis
2. **02_sentiment.ipynb** – Sentiment analysis with FinBERT
3. **03_feature.ipynb** – Feature engineering
4. **04_linreg.ipynb** – Model training & hyperparameter MultiOutputRegressor
5. **05_linreg_wo_sent.ipynb** – Model training & hyperparameter MultiOutputRegressor without Sentiment
6. **06_xgboost.ipynb** – Model training & hyperparameter SingleOutput and Early Stopping
7. **07_shap.ipynb** - Model evaluation & interpretability
8. **08_runners.ipynb** - Runners for all Models (Linear Regression, XGBoost, Random Forest, NN MLP, LSTM)

---

## ✅ Testing

Run all tests:
```bash
pytest
```
Only unit tests:
```bash
pytest -m "not integration"
```
Integration tests (needs `NEWS_API_KEY`):
```bash
pytest -m integration
```

---

## 📰 Ollama (Optional LLM)

Generate synthetic news for testing:
```bash
ollama list
ollama pull llama3
ollama run llama3
```
Set `.env`:
```
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

## 📊 Streamlit UI

Launch dashboard:
```bash
streamlit run app/ui/ui.py
```
Open <http://localhost:8501>

---

## 🐳 Docker Compose

Generate per-module requirements:
```bash
pipreqs app/api --force --savepath app/api/requirements.txt
pipreqs app/ui  --force --savepath app/ui/requirements.txt
pipreqs src     --force --savepath app/api/req-src.txt
```
Build & run:
```bash
docker compose build --no-cache # clean rebuild
docker compose up --build # reuse images
docker compose up -d # detached
```
Stop:
```bash
docker compose down
docker compose down -v # removes volumes
```
Status:
```bash
docker compose ps
```
Prune:
```bash
docker image prune
```
Ollama:
```bash
docker exec -it ollama ollama pull llama3 # first time only
docker exec -it ollama ollama list
```
Logs:
```bash
docker compose logs api --tail=200
docker compose logs ui --tail=200
```

---

## 📌 Notes

- Keep `.env` (with secrets like `NEWS_API_KEY`) **out of version control**.
- Commit a `.env.example` so others know what to set.
- Tail logs: `tail -f ollama.log uvicorn.log`

---

## 🤝 Contributing

-- Thanks for considering a contribution! This project welcomes issues, PRs, and ideas.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📝 TODO

- [ ] Implement **Random Forest Regressor**
- [ ] Implement **Neural Network (MLP)**
- [ ] Implement **LSTM** for sequential modeling
- [ ] Write **academic-style report** (thesis-like)
- [ ] Prepare **PowerPoint** presentation
- [ ] Deploy demo API + Streamlit to cloud (Render / Fly.io / AWS)
- [ ] If deployed:
    - [ ] Add authentication
    - [ ] Add rate-limiting
    - [ ] Add request logging
- [ ] Add monitoring & logging integration (Prometheus/Grafana)
- [ ] Add Model Tracking MLFlow
- [ ] Pipeline Orchestration: data -> train -> eval -> register -> deploy -> monitor
