import joblib
import pandas as pd
import streamlit as st


st.set_page_config(layout="wide", page_title="Stock Forecast")

st.title("Stock Forecasting App")
st.markdown("This app uses ML models to forecast log returns based on engineered features.")

model_bundle = joblib.load("../data/models/lin_reg_best.pkl.pkl")
model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]

def user_input_features():
    st.sidebar.header("Input Features")
    features = {
        "lag_5": st.sidebar.slider("Lag 5", -0.05, 0.05, 0.0, 0.001),
        "lag_10": st.sidebar.slider("Lag 10", -0.05, 0.05, 0.0, 0.001),
        "sma_5": st.sidebar.slider("SMA 5", -0.05, 0.05, 0.0, 0.001),
        "ema_10": st.sidebar.slider("EMA 10", -0.05, 0.05, 0.0, 0.001),
        "rsi": st.sidebar.slider("RSI", 0.0, 100.0, 50.0, 1.0),
        "macd": st.sidebar.slider("MACD", -1.0, 1.0, 0.0, 0.01),
        "quarter": st.sidebar.selectbox("Quarter", [1, 2, 3, 4]),
        "dow": st.sidebar.selectbox("Day of Week", list(range(7)))
    }

    return pd.DataFrame([features])

input_df = user_input_features()
X = preprocessor.transform(input_df)
pred = model.predict(X)
log_return = float(pred[0])

st.subheader("Predicted Log Return")
st.metric(label="Log Return", value=round(log_return, 5))

st.write("Input Features")
st.dataframe(input_df)
