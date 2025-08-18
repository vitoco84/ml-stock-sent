[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=vitoco84_ml-stock-sent&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=vitoco84_ml-stock-sent)

# Stock Market Forecast with FinBERT NLP Sentiment Analysis

## Configuration

* Upgrade pip: `python.exe -m pip install --upgrade pip`
* Install libraries: `pip install -r requirements.txt`
* Freeze libraries: `pip freeze > requirements.txt`
* Poetry: `poetry install`
* Check Nvidia Cuda Version: `nvidia-smi` -> 12.9
    * CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Pipeline Notebooks

* 01_eda.ipynb: EDA: Exploratory Data Analysis
* 02_sentiment.ipynb: Sentiment Analysis and Scoring with FinBERT
* 03_feature.ipynb: Feature Engineering
* 04_train.ipynb: Train / Test / Hyperparameter Tuning with Optuna
* 05_shap.ipynb: odel Evaluation

## Links and Datasets

* [DJIA News Kaggle Dataset](https://www.kaggle.com/datasets/aaron7sun/stocknews)
* [FinBERT NLP Model](https://huggingface.co/ProsusAI/finbert)
* [Stock Finance Data](https://finance.yahoo.com/)
  * [DJIA Example](https://finance.yahoo.com/quote/%5EDJI/history/)

## Tests

* Run PyTests from Terminal with: `pytest`

## FastAPI and Streamlit

* FastAPI
  * Run: `uvicorn app.api.main:app --reload`
  * Swagger: `http://localhost:8000/docs`

* Streamlist
  * Run: `streamlit run app/streamlit/ui.py`
  * Dashboar: `http://localhost:8501/`

* Or Run Script `.scripts/run_app.sh` from root folder in Terminal
* Logs: `tail -f ollama.log uvicorn.log`

### Endpoints

##### GET /

```json
{
  "message": "API is up and running!"
}
```

##### GET /price-history

* Params:
    * symbol: Ticker e.g.: AAPL, ^DJI
    * end_date: End date in YYYY-MM-DD
    * days: Lookback period

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
    },
    "..."
  ]
}
```

##### POST /predict-raw

* Request Body

```json
{
  "price": [
    {
      "date": "2020-01-01",
      "open": 100,
      "high": 105,
      "low": 99,
      "close": 104,
      "adj_close": 104,
      "volume": 1000000
    }
  ],
  "news": [
    {
      "date": "2020-01-01",
      "rank": "top1",
      "headline": "Apple stock surges today"
    },
    {
      "date": "2020-01-01",
      "rank": "top2",
      "headline": "Tech sector continues growth"
    }
  ]
}
```

```bash
curl -X POST http://localhost:8000/predict-raw \
 -H "Content-Type: application/json" \
 -d @tests/payload.json
```

* Response

```json
{
  "log_return": 0.0053,
  "current_price": 104.0,
  "predicted_price": 104.55
}
```

# Ollama Integration for News Generation and Testing

* Generating Fake News with Ollama
* List: `ollama list`
* Install model local: `ollama pull llama3`
* Run: `ollama run llama3`
* OLLAMA_URL = `http://localhost:11434/api/generate`