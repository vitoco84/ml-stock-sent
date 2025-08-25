from typing import List

import pandas as pd
import requests
from requests import RequestException

from src.logger import get_logger


logger = get_logger(__name__)

def generate_local_headlines(
        symbol: str,
        dates: List[str],
        url: str,
        model: str = "llama3",
        seed_examples: List[str] = None
) -> List[dict]:
    """Generate realistic financial headlines; uses seed_examples to guide tone/style."""
    logger.info(f"Generating {len(dates)} local headlines via LLM ({model}) for {symbol}")
    headlines: List[dict] = []

    seed_examples = seed_examples or []
    seed_examples = [s for s in seed_examples if isinstance(s, str) and s.strip()][:10]
    examples_block = ""
    if seed_examples:
        bullets = "\n".join(f"- {s.strip()}" for s in seed_examples)
        examples_block = (
            f"\nHere are example headlines about {symbol}:\n"
            f"{bullets}\n"
            "Follow the tone, specificity, and phrasing patterns above.\n"
        )

    for date in dates:
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        prompt = (
            f"You are a financial news editor. Write one realistic, concise headline for '{symbol}' dated {date_str}."
            f"{examples_block}"
            "Requirements:\n"
            "- ≤ 14 words; no emojis.\n"
            "- Do not copy the examples verbatim; keep plausibility.\n"
            "- Neutral to mildly analytical tone (Reuters/Bloomberg-like).\n"
            "Headline:"
        )

        try:
            response = requests.post(
                url,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "").strip()
            logger.info(f"Generated headline for {date_str}: {text}")
        except RequestException as e:
            text = f"{symbol} news on {date_str} (auto-generated)"
            logger.warning(f"Failed to generate headline for {date_str}: {e}")

        headlines.append({"date": date_str, "headline": text})

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
        raise ValueError("Enrich with LLM requires at least one real headline as an example.")

    real_news_df = pd.DataFrame(real_news)
    if "date" not in real_news_df.columns:
        raise ValueError("Missing 'date' in provided real_news records")
    real_news_df["date"] = pd.to_datetime(real_news_df["date"]).dt.strftime("%Y-%m-%d")

    real_dates = set(real_news_df["date"])
    missing_dates = sorted(set(price_dates) - real_dates)
    logger.info(f"Missing dates for LLM generation: {len(missing_dates)}")

    seed_examples = [str(x) for x in real_news_df["headline"].dropna().astype(str).tolist()][:20]

    generated_news = generate_local_headlines(
        symbol=symbol,
        dates=missing_dates,
        url=url_llm,
        model=model_llm,
        seed_examples=seed_examples
    ) if missing_dates else []

    enriched = real_news_df.to_dict(orient="records") + generated_news

    for row in enriched:
        if not isinstance(row.get("date"), str):
            row["date"] = pd.to_datetime(row.get("date")).strftime("%Y-%m-%d")

    enriched.sort(key=lambda x: x["date"])
    logger.info(f"Total enriched news rows: {len(enriched)}")
    return enriched
