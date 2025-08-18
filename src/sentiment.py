from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import Config
from src.logger import get_logger
from src.annotations import tested


class FinBERT:
    """FinBERT Class: Generate Sentiment scores and Embeddings."""

    def __init__(self, config: Config, device: str = "cuda", max_embedding_dims: int = None):
        self.config = config
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_embedding_dims = max_embedding_dims

        model_name = config.sentiment.model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True).to(
            self.device)
        self.embedder = self.classifier.base_model
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Loading FinBERT model: {model_name} on device: {self.device}")

    def _prepare_inputs(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_sentiment_scores(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        inputs = self._prepare_inputs(texts)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return [{"pos": p[0], "neu": p[1], "neg": p[2], "pos_minus_neg": p[0] - p[2]} for p in probs]

    def _get_embeddings(self, texts: Union[str, List[str]]) -> pd.Series:
        inputs = self._prepare_inputs(texts)
        with torch.no_grad():
            hidden = self.embedder(**inputs).last_hidden_state

        mean_embeddings = hidden.mean(dim=1).cpu().numpy()
        if self.max_embedding_dims is not None:
            mean_embeddings = mean_embeddings[:, :self.max_embedding_dims]
        return mean_embeddings

    def transform(self, df: pd.DataFrame, text_column: str = "headline", batch_size: int = 32) -> pd.DataFrame:
        df = df.copy()
        texts = df[text_column].tolist()

        sentiment_scores = []
        embeddings = []

        self.logger.info(f"Starting FinBERT transform on {len(texts)} texts with batch size {batch_size}")

        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Batch Processing"):
            batch = texts[i:i + batch_size]
            try:
                sentiment_scores += self._get_sentiment_scores(batch)
                embeddings.append(self._get_embeddings(batch))
            except Exception as e:
                self.logger.error(f"Batch {i}-{i + batch_size} failed: {e}")

        self.logger.info("FinBERT embedding and sentiment extraction complete.")

        sentiment_df = pd.DataFrame(sentiment_scores)
        emb_dim = embeddings[0].shape[1]
        embedding_df = pd.DataFrame(np.vstack(embeddings), columns=[f"emb_{i}" for i in range(emb_dim)])

        return pd.concat([df.reset_index(drop=True), sentiment_df, embedding_df], axis=1)

    @tested
    @staticmethod
    def aggregate_daily(df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        sentiment_cols = ["pos", "neu", "neg", "pos_minus_neg"]
        emb_cols = [col for col in df.columns if col.startswith("emb_")]

        agg_dict = {
            **dict.fromkeys(sentiment_cols, "mean"),
            **dict.fromkeys(emb_cols, "mean"),
            text_column: "count"
        }

        return (
            df.groupby("date")
            .agg(agg_dict)
            .fillna(0)
            .rename(columns={text_column: "headline_count"})
            .reset_index()
        )
