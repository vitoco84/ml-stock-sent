from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

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
    """
    Wrapper for FinBERT sentiment + embedding extraction.

    Features:
    - Sentiment scores (pos, neg, neu, pos_minus_neg)
    - Mean-pooled embeddings (optional truncated to `max_embedding_dims`)
    - Batch processing with caching (SHA256-based keys)
    - Deterministic torch execution if seed provided
    """

    def __init__(
            self,
            config: Config,
            device: str = "cuda",
            max_embedding_dims: int | None = None,
            cache_dir: str | Path = ".cache/finbert"
    ) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_embedding_dims = max_embedding_dims
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(self.__class__.__name__)

        self._set_deterministic(int(config.runtime.seed))

        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_FINBERT_MODEL)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            DEFAULT_FINBERT_MODEL,
            use_safetensors=True,
        ).to(self.device)
        self.embedder = self.classifier.base_model

        self.classifier.eval()
        self.embedder.eval()

        self.logger.info(f"Loaded FinBERT model: {DEFAULT_FINBERT_MODEL} on {self.device}")

        # Cache label mapping dynamically
        self.idx_map = self._resolve_label_indices()

    def _set_deterministic(self, seed: int) -> None:
        """Set torch seeds and deterministic algorithms where available."""
        try:
            torch.set_num_threads(1)
            torch.manual_seed(seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
        except Exception as e:
            self.logger.warning(f"Deterministic torch setup failed: {e}")

    def _resolve_label_indices(self) -> dict[str, int]:
        """Resolve FinBERT label indices dynamically (avoid hardcoding)."""
        mapping = {v.lower(): k for k, v in self.classifier.config.id2label.items()}
        return {
            "neu": mapping.get("neutral", 0),
            "pos": mapping.get("positive", 1),
            "neg": mapping.get("negative", 2)
        }

    def _hash_texts(self, texts: list[str]) -> str:
        """Stable cache key based on text order and content."""
        key = "\n".join(texts).encode("utf-8")
        return hashlib.sha256(key).hexdigest()

    def _cache_path(self, hash_key: str) -> Path:
        return self.cache_dir / f"{hash_key}.pkl"

    def _load_cache(self, cache_file: Path) -> dict[str, Any] | None:
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            cache_file.unlink(missing_ok=True)
            return None

    def _save_cache(self, cache_file: Path, result: dict[str, Any]) -> None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)  # type: ignore
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_file}: {e}")

    def _prepare_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _process_batch(self, texts: list[str]) -> dict[str, Any]:
        """Run FinBERT forward pass to get scores + embeddings."""
        inputs = self._prepare_inputs(texts)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            hidden = self.embedder(**inputs).last_hidden_state
            embeddings = hidden.mean(dim=1).cpu().numpy()

        scores = []
        for p in probs:
            scores.append(
                {
                    "neu": p[self.idx_map["neu"]],
                    "pos": p[self.idx_map["pos"]],
                    "neg": p[self.idx_map["neg"]],
                    "pos_minus_neg": p[self.idx_map["pos"]] - p[self.idx_map["neg"]],
                }
            )
        return {"scores": scores, "embeddings": embeddings}

    def _process_or_load_cache(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Check cache; if missing, process and save."""
        hash_key = self._hash_texts(texts)
        cache_file = self._cache_path(hash_key)
        cached = self._load_cache(cache_file)
        if cached is not None:
            return cached
        result = self._process_batch(texts)
        self._save_cache(cache_file, result)
        return result

    def transform(
            self,
            df: pd.DataFrame,
            text_column: str = "headline",
            batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Run FinBERT on a DataFrame of texts.

        Args:
            df: Input DataFrame
            text_column: Column name with text
            batch_size: Batch size for transformer

        Returns:
            DataFrame with sentiment + embedding columns
        """
        set_seed(self.config.runtime.seed)
        df = df.copy()
        texts = df[text_column].fillna("").astype(str).tolist()

        sentiment_scores: list[dict[str, float]] = []
        embedding_chunks: list[np.ndarray] = []

        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Batches"):
            batch = texts[i: i + batch_size]
            try:
                result = self._process_or_load_cache(batch)
                sentiment_scores.extend(result["scores"])
                embs = result["embeddings"]
                if self.max_embedding_dims:
                    embs = embs[:, : self.max_embedding_dims]
                embedding_chunks.append(embs)
            except Exception as e:
                self.logger.error(f"Batch {i}-{i + batch_size} failed: {e}")
                sentiment_scores.extend(
                    [{"neu": 1.0, "pos": 0.0, "neg": 0.0, "pos_minus_neg": 0.0}]
                    * len(batch)
                )
                zeros = np.zeros((len(batch), self.max_embedding_dims or 1))
                embedding_chunks.append(zeros)

        sentiment_df = pd.DataFrame(sentiment_scores)
        emb_dim = embedding_chunks[0].shape[1] if embedding_chunks else 0
        embedding_df = pd.DataFrame(
            np.vstack(embedding_chunks),
            columns=[f"emb_{i}" for i in range(emb_dim)],
        )
        return pd.concat([df.reset_index(drop=True), sentiment_df, embedding_df], axis=1)

    @staticmethod
    def aggregate_daily(df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
        """
        Aggregate sentiment + embeddings to daily level.
        - Mean across embeddings & sentiment
        - Count of headlines
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        sentiment_cols = ["pos", "neu", "neg", "pos_minus_neg"]
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

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
