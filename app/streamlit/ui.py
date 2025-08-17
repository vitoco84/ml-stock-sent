from datetime import datetime
import pandas as pd
import requests
import streamlit as st
import altair as alt
from pandas.tseries.offsets import BDay

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("Stock Prediction App (with FinBERT & LLM)")

# --- INPUTS ---
symbol = st.text_input("Ticker Symbol", value="AAPL")
end_date = st.date_input("End Date", value=datetime.today())
days = st.slider("Lookback Days", min_value=30, max_value=365, value=90)
enrich_flag = st.checkbox("Enrich with LLM if headlines missing", value=False)

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
        st.warning("Please fetch price history first.")
    else:
        payload = {
            "price": st.session_state.price_data,
            "news": news_input  # may be empty; LLM will handle if enrich=True
        }

        params = {"enrich": str(enrich_flag).lower()}
        with st.spinner("Predicting..."):
            response = requests.post(f"{API_BASE_URL}/predict-raw", params=params, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Complete")
            st.write(f"**Current Price:** ${result['current_price']:.2f}")
            st.write(f"**Predicted Price:** ${result['predicted_price']:.2f}")
            st.write(f"**Log Return:** {result['log_return']:.6f}")

            # ---- Build a chart with actual + predicted ----
            if "price_data" in st.session_state and st.session_state.price_data:
                price_df = pd.DataFrame(st.session_state.price_data).copy()
                price_df["date"] = pd.to_datetime(price_df["date"])
                price_df = price_df.sort_values("date")

                last_date = price_df["date"].max()
                pred_date = last_date + BDay(1)  # next business day
                predicted_price = float(result["predicted_price"])
                last_actual = float(price_df["adj_close"].iloc[-1])

                # Data for dashed segment from last actual to predicted
                segment_df = pd.DataFrame({
                    "date": [last_date, pred_date],
                    "price": [last_actual, predicted_price],
                    "layer": ["projection", "projection"]
                })

                # Actual line data
                actual_df = price_df.rename(columns={"adj_close": "price"})[["date", "price"]].copy()
                actual_df["layer"] = "actual"

                # Predicted point
                pred_point_df = pd.DataFrame({
                    "date": [pred_date],
                    "price": [predicted_price],
                    "layer": ["predicted"]
                })

                base = alt.Chart().encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("price:Q", title="Adj Close (USD)")
                )

                actual_line = base.mark_line().transform_filter(
                    alt.datum.layer == "actual"
                )

                projection_line = base.mark_line(strokeDash=[6, 6]).transform_filter(
                    alt.datum.layer == "projection"
                )

                pred_point = base.mark_point(size=70, color="red").transform_filter(
                    alt.datum.layer == "predicted"
                )

                pred_label = base.mark_text(dx=8, dy=-8, color="red").encode(
                    text=alt.Text("price:Q", format="$.2f")
                ).transform_filter(
                    alt.datum.layer == "predicted"
                )

                chart = alt.layer(
                    actual_line.data(actual_df),
                    projection_line.data(segment_df),
                    pred_point.data(pred_point_df),
                    pred_label.data(pred_point_df),
                ).properties(
                    width=700,
                    height=380,
                    title="Adj Close: Actual vs Predicted Next Business Day"
                )

                st.subheader("Price Chart")
                st.altair_chart(chart, use_container_width=True)
        else:
            try:
                error_msg = response.json().get("error", "Unknown error")
            except Exception:
                error_msg = response.text
            st.error(f"Prediction failed: {error_msg}")
