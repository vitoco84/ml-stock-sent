from __future__ import annotations

import re
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List

import pandas as pd
import requests
from requests import RequestException

from src.logger import get_logger


logger = get_logger(__name__)

def _generate_batch(symbol, batch, url, model, examples_block):
    """Generate headlines for a batch of dates (single API call)."""
    batch_str = "\n".join(f"- {pd.to_datetime(d).strftime('%Y-%m-%d')}" for d in batch)
    prompt = (
        f"You are a financial news editor. Write concise, realistic headlines for '{symbol}'.\n"
        f"Dates:\n{batch_str}\n"
        f"{examples_block}"
        "Requirements:\n"
        "- One headline per date.\n"
        "- ≤ 14 words; no emojis.\n"
        "- Neutral, analytical tone (Reuters/Bloomberg style).\n"
        "Output format:\n"
        "YYYY-MM-DD: headline\n"
        "YYYY-MM-DD: headline\n"
    )

    results: list[dict[str, str]] = []
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        # Parse lines like "YYYY-MM-DD: headline"
        for line in raw_text.splitlines():
            m = re.match(r"(\d{4}-\d{2}-\d{2})[:\- ]+(.*)", line.strip())
            if m:
                d, h = m.groups()
                h = h.strip()
                if len(h.split()) > 14:
                    h = " ".join(h.split()[:14])
                results.append({"date": d, "headline": h})

        # Fallback if nothing parsed
        if not results:
            for d in batch:
                date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
                results.append({"date": date_str, "headline": f"{symbol} news on {date_str} (auto-generated)"})

    except RequestException as e:
        logger.warning(f"Failed to generate headlines for batch {batch}: {e}")
        for d in batch:
            date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            results.append({"date": date_str, "headline": f"{symbol} news on {date_str} (auto-generated)"})

    return results

def generate_local_headlines(
        symbol: str,
        dates: List[str],
        url: str,
        model: str = "llama3",
        seed_examples: List[str] | None = None,
        batch_size: int = 10,
        max_workers: int = 4
) -> list[dict[str, str]]:
    """
    Generate realistic financial headlines using an LLM.
    - Batches dates to reduce API calls.
    - Runs batches in parallel for speed.
    """
    logger.info(f"Generating {len(dates)} local headlines via LLM ({model}) for {symbol}")
    headlines: list[dict[str, str]] = []

    # Prepare seed examples
    seed_examples = [s.strip() for s in (seed_examples or []) if isinstance(s, str) and s.strip()]
    seed_examples = seed_examples[:10]

    examples_block = ""
    if seed_examples:
        bullets = "\n".join(f"- {s}" for s in seed_examples)
        examples_block = (
            f"\nHere are example headlines about {symbol}:\n"
            f"{bullets}\n"
            "Follow the tone, specificity, and phrasing patterns above.\n"
        )

    # Split dates into batches
    batches = [dates[i: i + batch_size] for i in range(0, len(dates), batch_size)]

    # Run batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_batch, symbol, batch, url, model, examples_block): batch
            for batch in batches
        }
        for future in as_completed(futures):
            headlines.extend(future.result())

    return headlines

def enrich_news_with_generated(
        price_dates: List[str],
        real_news: list[dict[str, str]],
        symbol: str,
        url_llm: str,
        model_llm: str,
) -> list[dict[str, str]]:
    """
    Ensure every price date has at least one headline.
    Uses real news where available, fills gaps with generated headlines.
    """
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

    generated_news = (
        generate_local_headlines(
            symbol=symbol,
            dates=missing_dates,
            url=url_llm,
            model=model_llm,
            seed_examples=seed_examples,
        )
        if missing_dates
        else []
    )

    enriched = real_news_df.to_dict(orient="records") + generated_news

    # Normalize and deduplicate
    for row in enriched:
        if not isinstance(row.get("date"), str):
            row["date"] = pd.to_datetime(row.get("date").dt.strftime("%Y-%m-%d"))

    dedup: dict[str, dict[str, str]] = {
        str(row["date"]): {"date": str(row["date"]), "headline": str(row["headline"])}
        for row in enriched
    }

    enriched_list: list[dict[str, str]] = list(dedup.values())
    enriched_list.sort(key=lambda x: x["date"])

    logger.info(f"Total enriched news rows: {len(enriched_list)}")
    return enriched_list
