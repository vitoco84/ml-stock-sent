import os
import re
from datetime import datetime
from typing import Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Stock Prediction App", layout="centered")

API_URL = os.getenv("API_URL", "http://localhost:8000")
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT_FETCH = 15.0
READ_TIMEOUT_PREDICT = 180.0

@st.cache_resource
def http():
    s = requests.Session()
    s.headers.update({"Connection": "keep-alive"})
    return s

HTTP = http()

st.title("Stock Prediction App (FinBERT + LLM)")

def load_csv(file, date_col: str = "date") -> Optional[pd.DataFrame]:
    if file is None:
        return None
    df = pd.read_csv(file)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def clear_fetch_state():
    for k in ["symbol", "end_date", "days", "news_input", "fetched_price_df"]:
        st.session_state.pop(k, None)

def clear_csv_state():
    for k in ["price_csv_df", "news_csv_df"]:
        st.session_state.pop(k, None)

def symbol_valid(symbol: str):
    if not re.fullmatch(r"[A-Za-z0-9_.^-]+", symbol):
        st.error("Invalid symbol format.")
        st.stop()

mode = st.radio("Data source", ["Fetch from API", "Upload CSVs"], horizontal=True)

if mode == "Upload CSVs":
    clear_fetch_state()
    st.subheader("Upload CSVs")

    # Prices
    st.markdown("<span style='color:#16a34a; font-weight:700'>Prices CSV</span>", unsafe_allow_html=True)
    price_file = st.file_uploader(
        "Prices CSV (date, open, high, low, close, adj_close, volume)",
        type=["csv"],
        key="price_upl",
        label_visibility="collapsed",
    )

    # News
    st.markdown("<span style='color:#2563eb; font-weight:700'>News CSV (optional)</span>", unsafe_allow_html=True)
    news_file = st.file_uploader(
        "News CSV (date, headline)",
        type=["csv"],
        key="news_upl",
        label_visibility="collapsed",
    )

    # Keep session_state in sync with uploaders
    if price_file:
        st.session_state.price_csv_df = load_csv(price_file)
        st.success(f"Loaded {len(st.session_state.price_csv_df)} price rows")
    else:
        # If user clears the uploader, drop cached Data
        st.session_state.pop("price_csv_df", None)

    if news_file:
        news_df = load_csv(news_file)
        st.session_state.news_csv_df = news_df
        st.success(f"Loaded {len(news_df)} news rows")
    else:
        st.session_state.pop("news_csv_df", None)

    # Enrich flag
    st.checkbox(
        "Enrich with LLM if headlines missing",
        key="enrich_flag",
        value=False,
    )

    # Only enable Predict when prices are available
    can_predict_csv = (
            isinstance(st.session_state.get("price_csv_df"), pd.DataFrame)
            and not st.session_state["price_csv_df"].empty
    )
    predict_btn = st.button("Predict Price", disabled=not can_predict_csv)

else:
    clear_csv_state()
    with st.form("fetch_controls"):
        symbol = st.text_input("Ticker Symbol", value=st.session_state.get("symbol", "AAPL"))
        symbol_valid(symbol)
        end_date = st.date_input("End Date", value=st.session_state.get("end_date", datetime.today()))
        days = st.slider("Lookback Days", min_value=30, max_value=365, value=int(st.session_state.get("days", 90)))
        st.checkbox("Enrich with LLM if headlines missing", key="enrich_flag", value=False)

        st.subheader("Optional headlines for today")
        news_input = []
        for i in range(3):
            headline = st.text_input(f"Headline {i + 1}", key=f"headline_{i}")
            if headline:
                news_input.append({"date": end_date.strftime("%Y-%m-%d"), "headline": headline})

        c1, c2 = st.columns(2)
        with c1:
            fetch_btn = st.form_submit_button("Fetch Price History")
        with c2:
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

if predict_btn:
    if mode == "Upload CSVs":
        price_df = st.session_state.get("price_csv_df")
        if price_df is None or price_df.empty:
            st.warning("Upload a Prices CSV first.")
            st.stop()
        news_df = st.session_state.get("news_csv_df")
        news_records = news_df.to_dict(orient="records") if isinstance(news_df, pd.DataFrame) else []
        payload = {"price": price_df.to_dict(orient="records"), "news": news_records}
        params = {
            "enrich": st.session_state.get("enrich_flag", False),
            "horizon": 30,
            "return_path": True,
            "symbol": "CSV",
        }
    else:
        price_df = st.session_state.get("fetched_price_df")
        if price_df is None or price_df.empty:
            st.warning("Fetch price history first.")
            st.stop()
        news_records = st.session_state.get("news_input", [])
        payload = {"price": price_df.to_dict(orient="records"), "news": news_records}
        params = {
            "enrich": st.session_state.get("enrich_flag", False) if "enrich_flag" in st.session_state else False,
            "horizon": 30,
            "return_path": True,
            "symbol": st.session_state.get("symbol", "AAPL"),
        }

    with st.spinner("Predicting (this may take a while for large CSV/news)…"):
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
            st.error(
                "Prediction timed out while waiting for the server. Increase the read timeout or reduce CSV/news size."
            )
            st.stop()
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            body = None
            try:
                body = e.response.json() if e.response is not None else None
            except Exception:
                body = getattr(e.response, "text", None)
            st.error(f"Prediction failed [{status}]: {body or e}")
            st.stop()

    st.success("Prediction Complete")
    current_price = float(result.get("current_price", float("nan")))
    predicted_price = float(result.get("predicted_price", float("nan")))
    log_return = float(result.get("log_return", float("nan")))
    st.write(f"**Current Price:** ${current_price:.2f}")
    st.write(f"**Next-day Price (h=1):** ${predicted_price:.2f}")
    st.write(f"**Next-day Log Return:** {log_return:.6f}")

    df_prices = price_df.copy()

    if "adj_close" not in df_prices.columns or "date" not in df_prices.columns:
        st.info("Cannot plot history: 'date' or 'adj_close' missing in price data.")
        st.stop()

    df_prices["date"] = pd.to_datetime(df_prices["date"])
    df_prices = df_prices.sort_values("date")
    actual_df = df_prices.rename(columns={"adj_close": "price"})[["date", "price"]].copy()

    pred_dates = [pd.to_datetime(d) for d in list(result.get("predicted_dates", []))]
    pred_prices = [float(x) for x in list(result.get("predicted_price_path", []))]

    if pred_dates and pred_prices and len(pred_dates) == len(pred_prices):
        path_df = pd.DataFrame({"date": pred_dates, "price": pred_prices})
        x_enc, y_enc = alt.X("date:T", title="Date"), alt.Y("price:Q", title="Adj Close (USD)")
        chart = alt.layer(
            alt.Chart(actual_df).mark_line().encode(x_enc, y_enc),
            alt.Chart(path_df).mark_line(strokeDash=[6, 6]).encode(x_enc, y_enc),
            alt.Chart(path_df.tail(1)).mark_point(size=70, color="red").encode(x_enc, y_enc),
            alt.Chart(path_df.tail(1)).mark_text(dx=8, dy=-8, color="red").encode(
                x_enc, y_enc, text=alt.Text("price:Q", format="$.2f")
            ),
        ).properties(width=700, height=380, title="Adj Close: Actual + Predicted Next 30 Business Days")
        st.subheader("Price Chart")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No forecast path returned.")
