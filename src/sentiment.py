import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.config import Config
from src.logger import get_logger
from src.utils import set_seed


DEFAULT_FINBERT_MODEL = "yiyanghkust/finbert-tone"

class FinBERT:
    """FinBERT Class: Generate Sentiment scores and Embeddings with optional caching."""

    def __init__(
            self,
            config: Config,
            device: str = "cuda",
            max_embedding_dims: int = None,
            cache_dir: Union[str, Path] = ".cache/finbert"
    ):
        self.config = config
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_embedding_dims = max_embedding_dims

        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_FINBERT_MODEL)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            DEFAULT_FINBERT_MODEL,
            use_safetensors=True
        ).to(self.device)
        self.embedder = self.classifier.base_model

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Loaded FinBERT model: {DEFAULT_FINBERT_MODEL} on {self.device}")

        self.classifier.eval()
        self.embedder.eval()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_texts(self, texts: List[str]) -> str:
        key = "\n".join(sorted(texts)).encode("utf-8")
        return hashlib.sha256(key).hexdigest()

    def _cache_path(self, hash_key: str) -> Path:
        return self.cache_dir / f"{hash_key}.pkl"

    def _prepare_inputs(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _process_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        inputs = self._prepare_inputs(texts)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            hidden = self.embedder(**inputs).last_hidden_state
            embeddings = hidden.mean(dim=1).cpu().numpy()
        return {"scores": probs, "embeddings": embeddings}

    def _process_or_load_cache(self, texts: List[str]) -> Dict[str, np.ndarray]:
        hash_key = self._hash_texts(texts)
        cache_file = self._cache_path(hash_key)

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                self.logger.warning(f"Failed to load FinBERT cache: {cache_file}")
                cache_file.unlink(missing_ok=True)

        result = self._process_batch(texts)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            self.logger.warning(f"Failed to write FinBERT cache: {cache_file}")
        return result

    def transform(self, df: pd.DataFrame, text_column: str = "headline", batch_size: int = 32) -> pd.DataFrame:
        set_seed(self.config.runtime.seed)

        df = df.copy()
        texts = df[text_column].fillna("").astype(str).tolist()

        sentiment_scores: List[Dict[str, float]] = []
        embeddings_chunks: List[np.ndarray] = []

        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Batch Processing"):
            batch = texts[i:i + batch_size]
            if not batch:
                continue
            try:
                result = self._process_or_load_cache(batch)
                scores = result["scores"]
                embs = result["embeddings"]

                # finbert-tone label order: [neutral, positive, negative]
                sentiment_scores += [
                    {"neu": p[0], "pos": p[1], "neg": p[2], "pos_minus_neg": p[1] - p[2]} for p in scores
                ]

                if self.max_embedding_dims:
                    embs = embs[:, :self.max_embedding_dims]
                embeddings_chunks.append(embs)
            except Exception as e:
                self.logger.error(f"Batch {i}-{i + batch_size} failed: {e}")

        sentiment_df = pd.DataFrame(sentiment_scores)
        emb_dim = embeddings_chunks[0].shape[1]
        embedding_df = pd.DataFrame(np.vstack(embeddings_chunks), columns=[f"emb_{i}" for i in range(emb_dim)])

        return pd.concat([df.reset_index(drop=True), sentiment_df, embedding_df], axis=1)

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
