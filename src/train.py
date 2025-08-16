from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.base import TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.metrics import metrics
from src.models.base import Base


class ModelTrainer:
    def __init__(
            self,
            model: Base,
            name: str,
            config: Dict[str, Any],
            output_path: str = "data/models",
            preprocessor: Optional[TransformerMixin] = None
    ):
        self.model = model
        self.name = name
        self.config = config
        self.output_path = Path(output_path)
        self.preprocessor = preprocessor or StandardScaler()
        self.logger = get_logger(self.__class__.__name__)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.logger.info("Starting model training...")
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_val_scaled = self.preprocessor.transform(X_val) if X_val is not None else None
        if hasattr(self.model, "train"):
            self.model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X, y):
        self.logger.info("Evaluating model...")
        X_scaled = self.preprocessor.transform(X)
        preds = self.model.predict(X_scaled)
        return metrics(y, preds)

    def predict(self, X):
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict(X_scaled)

    def save(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "preprocessor": self.preprocessor
        }, str(self.output_path / f"{self.name}.pkl"))

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        return obj["model"], obj["preprocessor"]

    def track_mlflow(self, metrics: Dict[str, float], log_model: bool = True):
        self.logger.info("Tracking run with MLflow.")
        mlflow.set_experiment("stock_forecasting")
        with mlflow.start_run(run_name=self.name):
            mlflow.log_params(self.config)
            mlflow.log_metrics(metrics)
            model_path = self.output_path / f"{self.name}.pkl"
            if model_path.exists() and log_model:
                mlflow.log_artifact(str(model_path))

    def objective(self, trial: optuna.Trial, X, y, n_splits: int = 3):
        params = self.model.suggest_hyperparameters(trial)
        model = self.model.__class__(**params)
        self.logger.info(f"Running Optuna trial with params: {params}")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        val_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            preprocessor = self.preprocessor
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_val_scaled = preprocessor.transform(X_val)

            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            score = metrics(y_val, preds)["rmse"]
            val_scores.append(score)

        return np.mean(val_scores)
