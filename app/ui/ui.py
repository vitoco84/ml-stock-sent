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
CONNECT_TIMEOUT, READ_TIMEOUT_FETCH, READ_TIMEOUT_PREDICT = 10.0, 15.0, 180.0

st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("Stock Prediction App")

AVAILABLE_MODELS = ["linreg.pkl", "random_forest.pkl", "xgboost.pkl", "lstm.pkl"]
selected_model = st.selectbox("Select model", AVAILABLE_MODELS, index=0)

@st.cache_resource
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Connection": "keep-alive"})
    return s

HTTP = get_http()

def load_csv(file: Any, date_col: str = "date") -> Optional[pd.DataFrame]:
    if file is None:
        return None
    df = pd.read_csv(file)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def validate_prices(df: pd.DataFrame) -> None:
    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    if set(df.columns) != required:
        st.error(f"Invalid Prices CSV. Expected exactly: {sorted(required)}")
        st.stop()

def validate_news(df: pd.DataFrame) -> None:
    required = {"date", "headline"}
    if set(df.columns) != required:
        st.error(f"Invalid News CSV. Expected exactly: {sorted(required)}")
        st.stop()

def build_payload(price_df: pd.DataFrame, news_df: Optional[pd.DataFrame], ignore_news: bool, symbol: str):
    news_records = news_df.to_dict(orient="records") if news_df is not None else []
    return {
        "payload": {"price": price_df.to_dict(orient="records"), "news": news_records},
        "params": {"ignore_news": ignore_news, "horizon": 20, "return_path": True, "symbol": symbol},
    }

def call_api(payload: dict, params: dict):
    r = HTTP.post(
        f"{API_URL}/predict-raw",
        params=params,
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_PREDICT)
    )
    r.raise_for_status()
    return r.json()

def plot_results(price_df: pd.DataFrame, result: dict, current_price: float):
    df_prices = price_df.copy()
    df_prices["date"] = pd.to_datetime(df_prices["date"])
    actual_df = df_prices.rename(columns={"adj_close": "price"})[["date", "price"]]

    pred_dates = [pd.to_datetime(d) for d in result.get("predicted_dates", [])]
    logret_path = [float(x) for x in result.get("log_return_path", [])]
    pred_prices = (current_price * np.exp(np.cumsum(logret_path))).tolist() if logret_path else []

    if pred_dates and len(pred_dates) == len(pred_prices):
        path_df = pd.DataFrame({"date": pred_dates, "price": pred_prices})
        chart = alt.layer(
            alt.Chart(actual_df).mark_line(color="#1F618D").encode(x="date:T", y="price:Q"),
            alt.Chart(path_df).mark_line(strokeDash=[6, 6], color="#117A65").encode(x="date:T", y="price:Q"),
            alt.Chart(path_df.tail(1)).mark_point(size=70, color="red").encode(x="date:T", y="price:Q"),
            alt.Chart(path_df.tail(1)).mark_text(dx=8, dy=-8, color="red").encode(
                x="date:T", y="price:Q", text=alt.Text("price:Q", format="$.2f")
            ),
        ).properties(width=700, height=380, title="Adj Close Forecast – Next 20 Business Days")
        st.altair_chart(chart, use_container_width=True)

def show_results(result: dict, price_df: pd.DataFrame):
    st.subheader("Prediction Results")
    current_price = float(result.get("current_price", float("nan")))
    st.write(f"**Current Price:** ${current_price:.2f}")

    rows = []
    if "log_return_path" in result:
        logret_path = [float(x) for x in result["log_return_path"]]
        for h in [1, 5, len(logret_path)]:  # show 1, 5, last day
            if len(logret_path) >= h:
                cum_ret = np.sum(logret_path[:h])
                implied_price = current_price * np.exp(cum_ret)
                rows.append({
                    "Horizon": f"{h} days",
                    "Predicted Log Return": f"{cum_ret:.4f}",
                    "Implied Adj Close": f"${implied_price:,.2f}"
                })

    if rows:
        st.table(pd.DataFrame(rows))

    plot_results(price_df, result, current_price)

mode = st.radio(
    "Data source",
    ["Fetch from API", "Upload CSVs"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Fetch from API":
    # Reactive news toggle outside form
    use_news = st.radio("News usage:", ["Ignore news", "Use news"]) == "Use news"

    with st.form("fetch_form"):
        symbol = st.text_input("Ticker Symbol", "^DJI")
        if not re.fullmatch(r"[A-Za-z0-9_.^-]+", symbol):
            st.error("Invalid ticker symbol.")
            st.stop()

        end_date = st.date_input("End Date", datetime.today())
        days = st.slider("Lookback Days", 20, 365, 90)

        news_input: list[dict[str, str]] = []
        if use_news:
            for i in range(3):
                hline = st.text_input(f"Headline {i + 1}", key=f"headline_{i}")
                if hline:
                    news_input.append({"date": end_date.strftime("%Y-%m-%d"), "headline": hline})

        c1, c2 = st.columns([2, 1])
        submitted = c1.form_submit_button("Fetch & Predict", type="primary")
        clear_btn = c2.form_submit_button("Clear", type="secondary")

    if clear_btn:
        st.session_state.clear()
        st.rerun()

    if submitted:
        if use_news and not news_input:
            st.error("You selected 'Use news' but did not provide any headlines.")
            st.stop()

        with st.spinner("Fetching price history..."):
            try:
                r = HTTP.get(
                    f"{API_URL}/price-history",
                    params={"symbol": symbol, "end_date": end_date.strftime("%Y-%m-%d"), "days": int(days)},
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_FETCH)
                )
                r.raise_for_status()
                rows = r.json().get("price", [])
            except requests.RequestException as e:
                st.error(f"Failed to fetch price history: {e}")
                st.stop()

        if not rows:
            st.warning("No price data returned.")
            st.stop()

        price_df = pd.DataFrame(rows)
        st.subheader("Price History (tail)")
        st.dataframe(price_df.tail(10))

        req = build_payload(price_df, pd.DataFrame(news_input) if news_input else None, not use_news, symbol)
        req["params"]["model_name"] = selected_model
        with st.spinner("Running prediction..."):
            try:
                result = call_api(req["payload"], req["params"])
            except requests.RequestException as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
        show_results(result, price_df)

if mode == "Upload CSVs":
    # dynamic keys for clearing
    if "price_csv_key" not in st.session_state:
        st.session_state.price_csv_key = 0
    if "news_csv_key" not in st.session_state:
        st.session_state.news_csv_key = 0

    st.markdown("<h4>Prices CSV (required)</h4>", unsafe_allow_html=True)
    st.code(
        "date, open, high, low, close, adj_close, volume\n"
        "2024-10-01,100.25,101.30,99.80,100.95,100.95,1203400",
        language="csv"
    )
    price_file = st.file_uploader(
        "Upload prices CSV",
        type=["csv"],
        key=f"price_csv_{st.session_state.price_csv_key}",
        label_visibility="collapsed",
    )

    st.markdown("<h4>News CSV (optional)</h4>", unsafe_allow_html=True)
    st.code(
        "date, headline\n"
        "2024-10-01,Tech stocks rally as inflation cools",
        language="csv",
    )
    news_file = st.file_uploader(
        "Upload news CSV (optional)",
        type=["csv"],
        key=f"news_csv_{st.session_state.news_csv_key}",
        label_visibility="collapsed",
    )

    price_df = load_csv(price_file) if price_file else None
    news_df = load_csv(news_file) if news_file else None

    if price_df is not None:
        validate_prices(price_df)
        st.success(f"Loaded {len(price_df)} price rows")

    if news_df is not None:
        validate_news(news_df)
        st.success(f"Loaded {len(news_df)} news rows")

    c1, c2 = st.columns([2, 1])
    run_csv = c1.button("Run Prediction", type="primary", disabled=price_df is None)
    clear_csv = c2.button("Clear", type="secondary")

    if clear_csv:
        st.session_state.price_csv_key += 1
        st.session_state.news_csv_key += 1
        st.session_state.clear()
        st.rerun()

    if run_csv and price_df is not None:
        ignore_news = news_df is None
        req = build_payload(price_df, news_df, ignore_news, "CSV")
        req["params"]["model_name"] = selected_model
        with st.spinner("Running prediction..."):
            try:
                result = call_api(req["payload"], req["params"])
            except requests.RequestException as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
        show_results(result, price_df)
