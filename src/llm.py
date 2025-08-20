from typing import List

import pandas as pd
import requests

from src.logger import get_logger


logger = get_logger(__name__)

def generate_local_headlines(
        symbol: str,
        dates: List[str],
        url: str,
        model: str = "llama3"
) -> List[dict]:
    logger.info(f"Generating {len(dates)} local headlines via LLM ({model}) for {symbol}")
    headlines = []

    for date in dates:
        prompt = (
            f"Write a realistic financial news headline for '{symbol}' on {date}. "
            f"Make it sound like a headline from a business news site."
        )
        logger.info(f"Prompting LLM for {date}: {prompt}")

        try:
            response = requests.post(url, json={
                "model": model,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "").strip()
            logger.info(f"Generated headline for {date}: {text}")
        except Exception as e:
            text = f"{symbol} news on {date} (auto-generated)"
            logger.warning(f"Failed to generate headline for {date}: {e}")

        headlines.append({
            "date": date,
            "rank": "top1",
            "headline": text
        })

    return headlines

def enrich_news_with_generated(price_dates: List[str], real_news: List[dict], symbol: str, url_llm: str,
                               model_llm: str) -> List[dict]:
    logger.info("Enriching news with generated headlines (LLM)")

    price_dates = sorted(set(pd.to_datetime(price_dates).strftime("%Y-%m-%d")))

    if not real_news:
        logger.warning("No real news provided — generating headlines for all price dates.")
        real_news_df = pd.DataFrame(columns=["date", "rank", "headline"])
    else:
        real_news_df = pd.DataFrame(real_news)
        if "date" not in real_news_df.columns:
            raise ValueError("Missing 'date' in provided real_news records")
        real_news_df["date"] = pd.to_datetime(real_news_df["date"]).dt.strftime("%Y-%m-%d")

    real_dates = set(real_news_df["date"])
    missing_dates = sorted(set(price_dates) - real_dates)

    logger.info(f"Missing dates for LLM generation: {missing_dates[:3]}... total: {len(missing_dates)}")

    generated_news = generate_local_headlines(symbol, missing_dates, url_llm, model_llm)

    enriched = real_news_df.to_dict(orient="records") + generated_news

    for row in enriched:
        if isinstance(row["date"], pd.Timestamp):
            row["date"] = row["date"].strftime("%Y-%m-%d")
        elif isinstance(row["date"], str):
            continue
        else:
            raise ValueError(f"Unexpected date type in enriched row: {type(row['date'])}")

    enriched.sort(key=lambda x: x["date"])
    logger.info(f"Total enriched news rows: {len(enriched)}")
    return enriched
