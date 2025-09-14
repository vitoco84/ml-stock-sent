import os
import re
from datetime import datetime
from typing import Any, Optional

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


API_URL: str = os.getenv("API_URL", "http://localhost:8000")
CONNECT_TIMEOUT: float = 5.0
READ_TIMEOUT_FETCH: float = 15.0
READ_TIMEOUT_PREDICT: float = 180.0

st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("^DJI Stock Prediction App (FinBERT and LLM)")

@st.cache_resource
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Connection": "keep-alive"})
    return s

HTTP: requests.Session = get_http()

def load_csv(file: Any, date_col: str = "date") -> Optional[pd.DataFrame]:
    """Load a CSV file into a DataFrame and normalize the date column."""
    if file is None:
        return None
    df = pd.read_csv(file)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def clear_fetch_state() -> None:
    for k in ["symbol", "end_date", "days", "news_input", "fetched_price_df"]:
        st.session_state.pop(k, None)

def clear_csv_state() -> None:
    for k in ["price_csv_df", "news_csv_df"]:
        st.session_state.pop(k, None)

def validate_symbol(symbol: str) -> None:
    """Ensure ticker symbol matches allowed regex."""
    if not re.fullmatch(r"[A-Za-z0-9_.^-]+", symbol):
        st.error("Invalid symbol format.")
        st.stop()

# ---------------------------------------------------------
# UI: Mode Selection
# ---------------------------------------------------------
mode = st.radio("Data source", ["Fetch from API", "Upload CSVs"], horizontal=True)
st.caption(
    "Model is currently trained for **^DJI (Dow Jones)**. "
    "You can change the ticker, but predictions may be less accurate."
)

# ---------------------------------------------------------
# Mode: Upload CSVs
# ---------------------------------------------------------
if mode == "Upload CSVs":
    clear_fetch_state()
    st.subheader("Upload CSVs")

    # Price uploader
    st.markdown("**📈 Prices CSV**")
    price_file = st.file_uploader(
        "Prices CSV (date, open, high, low, close, adj_close, volume)",
        type=["csv"],
        key="price_upl",
        label_visibility="collapsed",
    )

    if price_file:
        st.session_state.price_csv_df = load_csv(price_file)
        st.success(f"Loaded {len(st.session_state.price_csv_df)} price rows")
    else:
        st.session_state.pop("price_csv_df", None)

    # News uploader
    st.markdown("**📰 News CSV (optional)**")
    news_file = st.file_uploader(
        "News CSV (date, headline)",
        type=["csv"],
        key="news_upl",
        label_visibility="collapsed",
    )

    if news_file:
        st.session_state.news_csv_df = load_csv(news_file)
        st.success(f"Loaded {len(st.session_state.news_csv_df)} news rows")
    else:
        st.session_state.pop("news_csv_df", None)

    # Validate price CSV columns
    price_df_preview = st.session_state.get("price_csv_df")
    if isinstance(price_df_preview, pd.DataFrame) and not price_df_preview.empty:
        required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        missing = required - set(price_df_preview.columns)
        if missing:
            st.warning(f"Prices CSV missing columns: {sorted(missing)}")

    # Missing-news strategy
    fill_strategy_csv = st.radio(
        "Missing-news strategy",
        ["Do nothing (ignore news)", "Enrich with LLM (needs ≥1)", "Pad with neutral (needs ≥2)"],
        index=0,
    )
    enrich_flag_csv = fill_strategy_csv == "Enrich with LLM (needs ≥1)"
    pad_neutral_csv = fill_strategy_csv == "Pad with neutral (needs ≥2)"
    ignore_news_csv = fill_strategy_csv == "Do nothing (ignore news)"

    can_predict_csv = isinstance(st.session_state.get("price_csv_df"), pd.DataFrame) and not st.session_state[
        "price_csv_df"
    ].empty
    predict_btn = st.button("Predict Price", disabled=not can_predict_csv)

# ---------------------------------------------------------
# Mode: Fetch from API
# ---------------------------------------------------------
else:
    clear_csv_state()
    with st.form("fetch_controls"):
        symbol = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("symbol", "^DJI"),
            help="Prefilled with ^DJI (Dow Jones). You can change it if needed.",
        )
        validate_symbol(symbol)

        end_date = st.date_input("End Date", value=st.session_state.get("end_date", datetime.today()))
        c1, c2 = st.columns([1, 1])
        with c1:
            days = st.slider("Lookback Days", min_value=20, max_value=365, value=int(st.session_state.get("days", 90)))
        with c2:
            st.markdown("**Forecast horizon: fixed 20 business days**")

        fill_strategy = st.radio(
            "Missing-news strategy",
            ["Do nothing (ignore news)", "Enrich with LLM (needs ≥1)", "Pad with neutral (needs ≥2)"],
            index=0,
        )
        enrich_flag = fill_strategy == "Enrich with LLM (needs ≥1)"
        pad_neutral_flag = fill_strategy == "Pad with neutral (needs ≥2)"
        ignore_news_flag = fill_strategy == "Do nothing (ignore news)"

        st.subheader("Optional headlines for today")
        news_input: list[dict[str, str]] = []
        for i in range(3):
            headline = st.text_input(f"Headline {i + 1}", key=f"headline_{i}")
            if headline:
                news_input.append({"date": end_date.strftime("%Y-%m-%d"), "headline": headline})

        c1b, c2b = st.columns(2)
        with c1b:
            fetch_btn = st.form_submit_button("Fetch Price History")
        with c2b:
            predict_btn = st.form_submit_button("Predict Price")

    if fetch_btn:
        with st.spinner("Fetching price history..."):
            try:
                r = HTTP.get(
                    f"{API_URL}/price-history",
                    params={"symbol": symbol, "end_date": end_date.strftime("%Y-%m-%d"), "days": int(days)},
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_FETCH),
                )
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as e:
                st.session_state.pop("fetched_price_df", None)
                st.error(f"API request failed: {e}")
            else:
                rows = data.get("price", [])
                if rows:
                    st.session_state.fetched_price_df = pd.DataFrame(rows)
                    st.session_state.symbol = symbol
                    st.session_state.end_date = end_date
                    st.session_state.days = int(days)
                    st.session_state.news_input = news_input
                    st.subheader("Price History (tail)")
                    st.dataframe(st.session_state.fetched_price_df.tail(10))
                else:
                    st.session_state.pop("fetched_price_df", None)
                    st.warning("No price data returned.")

# ---------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------
if "predict_btn" in locals() and predict_btn:
    # Prepare payload
    if mode == "Upload CSVs":
        price_df = st.session_state.get("price_csv_df")
        if price_df is None or price_df.empty:
            st.warning("Upload a Prices CSV first.")
            st.stop()

        required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        missing = required - set(price_df.columns)
        if missing:
            st.error(f"Prices CSV missing columns: {sorted(missing)}")
            st.stop()

        news_df = st.session_state.get("news_csv_df")
        if enrich_flag_csv and (news_df is None or news_df.empty):
            st.error("Enrich requires ≥1 headline in News CSV.")
            st.stop()
        if pad_neutral_csv and (news_df is None or len(news_df) < 2):
            st.error("Pad with neutral requires ≥2 headlines in News CSV.")
            st.stop()

        news_records: list[dict[str, str]] = (
            [{str(k): str(v) for k, v in row.items()} for row in news_df.to_dict(orient="records")]
            if isinstance(news_df, pd.DataFrame)
            else []
        )
        payload = {"price": price_df.to_dict(orient="records"), "news": news_records}
        params = {
            "enrich": enrich_flag_csv,
            "pad_neutral": pad_neutral_csv,
            "ignore_news": ignore_news_csv,
            "horizon": 20,
            "return_path": True,
            "symbol": "CSV",
        }

    else:
        price_df = st.session_state.get("fetched_price_df")
        if price_df is None or price_df.empty:
            st.warning("Fetch price history first.")
            st.stop()

        news_records: list[dict[str, str]] = [
            {str(k): str(v) for k, v in row.items()} for row in st.session_state.get("news_input", [])
        ]
        payload = {"price": price_df.to_dict(orient="records"), "news": news_records}
        params = {
            "enrich": enrich_flag,
            "pad_neutral": pad_neutral_flag,
            "ignore_news": ignore_news_flag,
            "horizon": 20,
            "return_path": True,
            "symbol": st.session_state.get("symbol", "^DJI"),
        }

    # Call API
    with st.spinner("Predicting (may take a while for large CSV/news)…"):
        try:
            r = HTTP.post(
                f"{API_URL}/predict-raw",
                params=params,
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_PREDICT),
            )
            r.raise_for_status()
            result = r.json()
        except requests.ReadTimeout:
            st.error("Prediction timed out. Try smaller input or increase timeout.")
            st.stop()
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            body = getattr(getattr(e, "response", None), "text", None)
            st.error(f"Prediction failed [{status}]: {body or e}")
            st.stop()

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    st.success("Prediction Complete")
    current_price = float(result.get("current_price", float("nan")))
    st.write(f"**Current Price:** ${current_price:.2f}")

    rows = []

    # Rolling mode: detect keys like log_return_5, log_return_20...
    rolling_keys = [k for k in result.keys() if k.startswith("log_return_")]
    if rolling_keys:
        for k in sorted(rolling_keys, key=lambda x: int(x.split("_")[2])):
            h = int(k.split("_")[2])
            log_ret = float(result[k])
            implied_price = current_price * np.exp(log_ret)
            rows.append({
                "Horizon": f"{h} days",
                "Predicted Log Return": f"{log_ret:.4f}",
                "Implied Adj Close": f"${implied_price:,.2f}"
            })

    # Step mode: cumulative from log_return_path
    elif "log_return_path" in result:
        logret_path = [float(x) for x in result.get("log_return_path", [])]
        for h in [1, len(logret_path)]:  # you could also show multiple points
            if len(logret_path) >= h:
                cum_ret = np.sum(logret_path[:h])
                implied_price = current_price * np.exp(cum_ret)
                rows.append({
                    "Horizon": f"{h} days",
                    "Predicted Log Return": f"{cum_ret:.4f}",
                    "Implied Adj Close": f"${implied_price:,.2f}"
                })

    # Show results
    if rows:
        st.subheader("Predicted Return & Implied Price")
        st.table(pd.DataFrame(rows))

    # ---------------------------------------------------------
    # Plot forecast path
    # ---------------------------------------------------------
    df_prices = price_df.copy()
    if {"adj_close", "date"} <= set(df_prices.columns):
        df_prices["date"] = pd.to_datetime(df_prices["date"])
        actual_df = df_prices.rename(columns={"adj_close": "price"})[["date", "price"]].copy()

        pred_dates = [pd.to_datetime(d) for d in result.get("predicted_dates", [])]
        logret_path = [float(x) for x in result.get("log_return_path", [])]

        pred_prices = (current_price * np.exp(np.cumsum(logret_path))).tolist() if logret_path else []

        if pred_dates and pred_prices and len(pred_dates) == len(pred_prices):
            path_df = pd.DataFrame({"date": pred_dates, "price": pred_prices})
            chart = alt.layer(
                alt.Chart(actual_df).mark_line().encode(x="date:T", y="price:Q"),
                alt.Chart(path_df).mark_line(strokeDash=[6, 6]).encode(x="date:T", y="price:Q"),
                alt.Chart(path_df.tail(1)).mark_point(size=70, color="red").encode(x="date:T", y="price:Q"),
                alt.Chart(path_df.tail(1))
                .mark_text(dx=8, dy=-8, color="red")
                .encode(x="date:T", y="price:Q", text=alt.Text("price:Q", format="$.2f")),
            ).properties(
                width=700,
                height=380,
                title="Adj Close Forecast – Next 20 Business Days",
            )
            st.subheader("Price Chart")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No forecast path returned.")
    else:
        st.info("Cannot plot history: missing 'date' or 'adj_close'.")
