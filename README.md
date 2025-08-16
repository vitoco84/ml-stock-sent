# Stock Market Forecast with FinBERT NLP Sentiment Analysis

## Configuration

* Upgrade pip: `python.exe -m pip install --upgrade pip`
* Install libraries: `pip install -r requirements.txt`
* Freeze libraries: `pip freeze > requirements.txt`
* Check Nvidia Cuda Version: `nvidia-smi` -> 12.9
  * CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Pipeline

* EDA: Exploratory Data Analysis
* Sentiment Analysis and Scoring with FinBERT
* Feature Engineering
* Train / Test
* Hyperparameter Tuning with Optuna
* Model Evaluation

## Links and Datasets

* [DJIA News Kaggle Dataset](https://www.kaggle.com/datasets/aaron7sun/stocknews)
* [FinBERT NLP Model](https://huggingface.co/ProsusAI/finbert)

## Tests

* Run PyTests from Terminal with: `pytest`