# 📈 Stock Market Forecast with FinBERT Sentiment Analysis

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=vitoco84_ml-stock-sent&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=vitoco84_ml-stock-sent)

Forecast stock prices by combining:
- 📊 Historical market data (Yahoo Finance)
- 📰 News sentiment via [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)
- 🤖 Optional headline generation with local LLMs (Ollama)
- ⚡ A production-ready FastAPI backend and Streamlit dashboard

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10+**
- (Optional) NVIDIA GPU with CUDA for accelerated FinBERT and training/tuning

### Setup
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Verify CUDA (optional)
```bash
# confirm driver and CUDA runtime
nvidia-smi
# Example: install PyTorch for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📂 Project Structure

- `src/` - models, data processing, feature engineering, model training
- `app/api/` - FastAPI application
- `app/ui/` - Streamlit dashboard
- `notebooks/` - Jupyter notebooks for exploration and training
- `data/` - figures, processed, models, raw
- `config/` - configuration
- `scripts/` - freeze requirements

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
# External news provider (optional if you won’t call /news-history)
NEWS_API_BASE=your_news_apy_base_url
NEWS_API_KEY=your_newsapi_key

# Optional LLM for news enrichment (used when enrich=true)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# API runtime
API_ROOT_PATH=/
CORS_ORIGINS=["http://localhost:8501","http://localhost:3000"]
```

> Note: The API checks reachability when enrich=true. Ensure OLLAMA_BASE is live (native or Docker).

---

## 📡 API Endpoints and Schemas

### `GET /healthz`
**Response**
```json
{ "ok": true }
```

---

### `GET /price-history`
Fetch historical stock data.

**Query params**
- `symbol` *(str, default ^DJI)*
- `end_date` *(YYYY-MM-DD, required)*
- `days` *(int, default 90)* - business days to look back

**Response - `PriceHistoryResponse`**
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
- `400` if lookback > 365 business days
- `404` if no data for symbol/date range
- `500` on unexpected failure

---

### `GET /news-history`
Fetch recent news headlines via NewsAPI.

**Query params**
- `query` *(str, required)* - search term, e.g. `Apple`
- `end_date` *(YYYY-MM-DD, required)*
- `days` *(int, default 7)* - lookback window

**Success - `NewsHistoryResponse`**
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
- `400` if lookback > 365 business days
- `500` with message `"Missing NEWS_API_KEY environment variable"` if the key is not configured
- `500` on unexpected failure

---

### `POST /predict-raw`
Predict next price log-returns using historical prices, news sentiment, and FinBERT.

**Query params**
- `symbol` *(str, required)* - Ticker symbol for context (e.g., `AAPL`)
- `return_path` *(bool, default `true`)* - whether to return full H‑step paths
- `enrich` *(bool, default `false`)* - generate missing headlines locally (requires reachable `OLLAMA_URL`)
- `pad_neutral` *(bool, default `false`)* - Use provided headlines and fill missing days with neutral sentiment (requires ≥2 headlines, no generation)
- `ignore_news` *(bool, default `false`)* - Ignore all news (neutral sentiment every day)

> Note: Choose exactly one strategy: ignore_news OR enrich OR pad_neutral.

**Request body - `PredictionRequest`**
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

**Response - `PredictionResponse` (Step Target Mode, when `return_path=true`)**
```json
{
  "horizon": 20,
  "current_price": 105.0,
  "log_return": 0.0035,
  "predicted_price": 105.37,
  "log_return_path": [0.0035, 0.0012, 0.0007, -0.0003, 0.0021],
  "predicted_price_path": [105.37, 105.50, 105.57, 105.54, 105.77],
  "predicted_dates": ["2025-01-10", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16"],
  "last_date": "2025-01-09"
}
```
**Response - `PredictionResponse` (Rolling Target Mode, when `return_path=true`)**
```json
{
  "horizon": [1, 5, 20],
  "current_price": 105.0,
  "log_return_1": -0.0021,
  "predicted_price_1": 104.78,
  "log_return_5": 0.0065,
  "predicted_price_5": 105.69,
  "log_return_20": 0.0123,
  "predicted_price_20": 106.29,
  "log_return": -0.0021,
  "predicted_price": 106.29,
  "log_return_path": [-0.0001, -0.0001, ...], 
  "predicted_price_path": [105.8, 105.9, ...],
  "predicted_dates": ["2025-01-10", "2025-01-13", "..."],
  "last_date": "2025-01-09"
}
```
**Response - `PredictionResponse` (when `return_path=false`)**
```json
{
  "horizon": 5,
  "log_return": 0.0035,
  "current_price": 105.0,
  "predicted_price": 105.37
}
```

**Validation and errors**
- `400` if price or news rows > 2000, span > 5 years, or invalid flag combination
- `422` if `price` is missing/empty, malformed payloads, or pad_neutral with <2 headlines
- `500` on feature generation/model errors or enrichment failure

**cURL example**
```bash
curl -X POST "http://localhost:8000/predict-raw?symbol=AAPL&return_path=true&ignore_news=false&enrich=false&pad_neutral=true" \
  -H "Content-Type: application/json" \
  -d '{
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

1. **01_eda.ipynb** - Exploratory Data Analysis
2. **02_sentiment.ipynb** - Sentiment analysis with FinBERT
3. **03_feature.ipynb** - Feature engineering
4. **04_pipeline.ipynb** - Train and Tune Models (linreg, xgboost, random_forest, cnn and lstm)
5. **05_eval.ipynb** - Stationarity test (Augmented Dickey–Fuller) and interpretability (SHAP)
6. **06_results_plots.ipynb** - Combined Plots and Results

> Note: The Notebooks should be run in order.

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
Tests with coverage HTML report
```bash
pytest --cov=src --cov-report=html
```

---

## 📰 Ollama (Optional LLM)

Generate synthetic news for testing:

Local (native)
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

Docker Compose (service-to-service)
```
# The API resolves Ollama by service name
OLLAMA_BASE=http://ollama:11434
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
Build and run:
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
docker ps -a
docker compose ps
```
Prune:
```bash
docker image prune
docker image prune -af # -af remove all unused and skip confirmation
```
Ollama:
```bash
docker pull ollama/ollama:latest

# Run docker compose up then pull model
docker exec -it ollama ollama pull llama3 # first time only pull ollama llama3
docker exec -it ollama ollama list # list installed models
```
Logs:
```bash
docker compose logs -f api
docker compose logs -f ui
```

---

## 📌 Notes

- Keep `.env` (with secrets like `NEWS_API_KEY`) **out of version control**.
- Tail logs: `tail -f ollama.log uvicorn.log`

---

## 🤝 Contributing

- Thanks for considering a contribution! Issues, PRs, and ideas are welcome.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📝 TODO

- [ ] Transfer Learning and Fine-Tuning
- [ ] Write **academic-style report** (thesis-like)
- [ ] Prepare **PowerPoint** presentation
- [ ] Optional: Deploy demo API and Streamlit to cloud (Render / Fly.io / AWS)
- [ ] If deployed:
    - [ ] Add authentication
    - [ ] Add rate-limiting
    - [ ] Add request logging
- [ ] Optional: Add monitoring and logging integration (Prometheus/Grafana)
- [ ] Optional: Add Model Tracking MLFlow
- [ ] Optional: Pipeline Orchestration: data -> train -> eval -> register -> deploy -> monitor
