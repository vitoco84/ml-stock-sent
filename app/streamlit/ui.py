from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("Stock Prediction App (with FinBERT & LLM)")

# --- INPUTS ---
symbol = st.text_input("Ticker Symbol", value="AAPL")
end_date = st.date_input("End Date", value=datetime.today())
days = st.slider("Lookback Days", min_value=30, max_value=365, value=90)
enrich_flag = st.checkbox("Enrich with LLM if headlines missing", value=False)

st.subheader("Upload CSVs (optional)")
price_file = st.file_uploader("Prices CSV (date, open, high, low, close, adj_close, volume)", type=["csv"])
news_file = st.file_uploader("News CSV (date, rank, headline)", type=["csv"])

if price_file is not None:
    df_up = pd.read_csv(price_file)
    if "date" in df_up.columns:
        df_up["date"] = pd.to_datetime(df_up["date"]).dt.strftime("%Y-%m-%d")
    st.session_state.price_data = df_up.to_dict(orient="records")
    st.success(f"Loaded {len(df_up)} price rows from CSV")

if news_file is not None:
    df_news_up = pd.read_csv(news_file)
    if "date" in df_news_up.columns:
        df_news_up["date"] = pd.to_datetime(df_news_up["date"]).dt.strftime("%Y-%m-%d")
    st.session_state.news_data = df_news_up.to_dict(orient="records")
    st.success(f"Loaded {len(df_news_up)} news rows from CSV")

# --- FETCH PRICE DATA ---
if st.button("Fetch Price History"):
    with st.spinner("Fetching price history..."):
        response = requests.get(
            f"{API_BASE_URL}/price-history",
            params={"symbol": symbol, "end_date": end_date.strftime("%Y-%m-%d"), "days": days}
        )

        if response.status_code == 200:
            data = response.json()
            if "price" in data:
                st.session_state.price_data = data["price"]
                price_df = pd.DataFrame(data["price"])
                st.subheader("Price History")
                st.dataframe(price_df.tail(10))
            else:
                st.error(data.get("error", "Unknown error while fetching price history."))
        else:
            st.error("API request failed. Check server logs or connection.")

# --- NEWS HEADLINES INPUT ---
st.subheader("News Headlines")
st.markdown("Optional: Enter 1–3 news headlines for today. If none are provided, LLM enrichment can fill the rest.")

news_input = []
for i in range(3):
    headline = st.text_input(f"Headline {i + 1}", key=f"headline_{i}")
    if headline:
        news_input.append({
            "date": end_date.strftime("%Y-%m-%d"),
            "rank": f"top{i + 1}",
            "headline": headline
        })

# --- PREDICTION ---
if st.button("Predict Price"):
    if "price_data" not in st.session_state or not st.session_state.price_data:
        st.warning("Please fetch price history or upload a prices CSV first.")
    else:
        payload = {
            "price": st.session_state.price_data,
            "news": st.session_state.get("news_data", news_input)  # prefer uploaded; else text inputs
        }
        params = {
            "enrich": enrich_flag,
            "horizon": 30,
            "return_path": True,
            "symbol": symbol,
        }

        with st.spinner("Predicting..."):
            response = requests.post(f"{API_BASE_URL}/predict-raw", params=params, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Complete")
            st.write(f"**Current Price:** ${result['current_price']:.2f}")
            st.write(f"**Next-day Price (h=1):** ${result['predicted_price']:.2f}")
            st.write(f"**Next-day Log Return:** {result['log_return']:.6f}")

            # ---- Build a chart with history + 30-day forecast path ----
            if "price_data" in st.session_state and st.session_state.price_data:
                price_df = pd.DataFrame(st.session_state.price_data).copy()
                price_df["date"] = pd.to_datetime(price_df["date"])
                price_df = price_df.sort_values("date")

                actual_df = price_df.rename(columns={"adj_close": "price"})[["date", "price"]].copy()
                actual_df["layer"] = "actual"

                pred_dates = result.get("predicted_dates", [])
                pred_prices = result.get("predicted_price_path", [])
                if hasattr(pred_dates, "tolist"): pred_dates = pred_dates.tolist()
                if hasattr(pred_prices, "tolist"): pred_prices = pred_prices.tolist()
                pred_dates = [pd.to_datetime(d) for d in list(pred_dates)]
                pred_prices = [float(x) for x in list(pred_prices)]

                if pred_dates and pred_prices and len(pred_dates) == len(pred_prices):
                    path_df = pd.DataFrame({"date": pred_dates, "price": pred_prices})
                    path_df["layer"] = "forecast"

                    encoding = alt.X("date:T", title="Date"), alt.Y("price:Q", title="Adj Close (USD)")

                    actual_line = alt.Chart(actual_df).mark_line().encode(*encoding).transform_filter(
                        alt.datum.layer == "actual"
                    )
                    forecast_line = alt.Chart(path_df).mark_line(strokeDash=[6, 6]).encode(*encoding).transform_filter(
                        alt.datum.layer == "forecast"
                    )
                    last_pt = path_df.tail(1)
                    pred_point = alt.Chart(last_pt).mark_point(size=70, color="red").encode(*encoding)
                    pred_label = alt.Chart(last_pt).mark_text(dx=8, dy=-8, color="red").encode(
                        *encoding, text=alt.Text("price:Q", format="$.2f")
                    )

                    chart = alt.layer(actual_line, forecast_line, pred_point, pred_label).properties(
                        width=700, height=380, title="Adj Close: Actual + Predicted Next 30 Business Days"
                    )
                    st.subheader("Price Chart")
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No forecast path returned; check API parameters.")
        else:
            try:
                error_msg = response.json().get("error", "Unknown error")
            except Exception:
                error_msg = response.text
            st.error(f"Prediction failed: {error_msg}")
