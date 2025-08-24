from typing import List

import pandas as pd
import requests
from requests import RequestException

from src.logger import get_logger


logger = get_logger(__name__)

def generate_local_headlines(symbol: str, dates: List[str], url: str, model: str = "llama3") -> List[dict]:
    """Prompts local Ollama to generate fake news headlines."""
    logger.info(f"Generating {len(dates)} local headlines via LLM ({model}) for {symbol}")
    headlines: List[dict] = []

    for date in dates:
        # Ensure YYYY-MM-DD
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")

        prompt = (
            f"Write a realistic financial news headline for '{symbol}' on {date_str}. "
            f"Make it sound like a headline from a business news site."
        )

        logger.info(f"Prompting LLM for {date_str}: {prompt}")

        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "").strip()
            logger.info(f"Generated headline for {date_str}: {text}")
        except RequestException as e:
            text = f"{symbol} news on {date_str} (auto-generated)"
            logger.warning(f"Failed to generate headline for {date_str}: {e}")

        headlines.append({
            "date": date_str,
            "headline": text
        })

    return headlines

def enrich_news_with_generated(
        price_dates: List[str],
        real_news: List[dict],
        symbol: str,
        url_llm: str,
        model_llm: str
) -> List[dict]:
    logger.info("Enriching news with generated headlines (LLM)")
    price_dates = sorted(set(pd.to_datetime(price_dates).strftime("%Y-%m-%d")))

    if not real_news:
        logger.warning("No real news provided — generating headlines for all price dates.")
        real_news_df = pd.DataFrame(columns=["date", "headline"])
    else:
        real_news_df = pd.DataFrame(real_news)
        if "date" not in real_news_df.columns:
            raise ValueError("Missing 'date' in provided real_news records")
        real_news_df["date"] = pd.to_datetime(real_news_df["date"]).dt.strftime("%Y-%m-%d")

    real_dates = set(real_news_df["date"])
    missing_dates = sorted(set(price_dates) - real_dates)
    logger.info(f"Missing dates for LLM generation: {missing_dates[:3]}... total: {len(missing_dates)}")

    generated_news = generate_local_headlines(symbol, missing_dates, url_llm, model_llm) if missing_dates else []
    enriched = real_news_df.to_dict(orient="records") + generated_news

    for row in enriched:
        if not isinstance(row.get("date"), str):
            row["date"] = pd.to_datetime(row.get("date")).strftime("%Y-%m-%d")

    enriched.sort(key=lambda x: x["date"])
    logger.info(f"Total enriched news rows: {len(enriched)}")
    return enriched
