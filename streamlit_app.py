"""Chat and charts for the three ADMIE datasets."""
import os
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="GridMind GR", page_icon="⚡", layout="wide")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
st.title("⚡ GridMind GR")
st.caption("Historical net system load without Crete, renewable production, generation by source and load forecasts.")


def render(result, widget_key="direct"):
    st.markdown(result["answer"])
    data = result.get("data", [])
    if data:
        frame = pd.DataFrame(data)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if "technology" in frame:
            chart = frame.pivot_table(index="timestamp", columns="technology", values="production_mwh", aggfunc="sum")
            st.area_chart(chart, y_label="MWh per interval")
            st.bar_chart(frame.groupby("technology")["production_mwh"].sum(), y_label="MWh")
        else:
            value = next(c for c in ["load_mwh", "res_mwh", "forecast_mwh"] if c in frame)
            columns = [value]
            if {"lower_mwh", "upper_mwh"}.issubset(frame.columns):
                columns += ["lower_mwh", "upper_mwh"]
                st.caption("Lower and upper lines show historical error bounds, not guaranteed future limits.")
            st.line_chart(frame.set_index("timestamp")[columns], y_label="MWh per interval")
        with st.expander("View data"):
            st.dataframe(frame, hide_index=True)
        st.download_button("Download CSV", frame.to_csv(index=False), "gridmind.csv", "text/csv", key=f"csv-{widget_key}")
    if result.get("forecast_metadata"):
        with st.expander("Forecast method and historical validation"):
            st.json(result["forecast_metadata"])
    if result.get("sources"):
        st.caption("Sources: " + "; ".join(result["sources"]))


def call_api(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=180, **kwargs)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(str(detail))
            return None
        return response.json()
    except requests.RequestException:
        st.error("Cannot reach the API. Start the backend and check API_BASE_URL.")
        return None


page = st.sidebar.radio("Explore", ["Chat", "Historical data", "Load forecast", "Health"])
if page == "Chat":
    st.caption("Try: 'Show generation from 2026-01-01 to 2026-01-31', 'RES yesterday', or 'Forecast load tomorrow'.")
    if st.button("New chat"):
        st.session_state["messages"] = []
        st.session_state.pop("chat_context", None)
        st.rerun()
    st.caption("Follow up with 'and the next day?', 'show renewables instead', or 'what was the peak?'.")
    messages = st.session_state.setdefault("messages", [])
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Separate widget identities when a repeated question has the same answer.
                with st.container(key=f"message-{i}"):
                    render(message["result"], widget_key=str(i))
            else:
                st.markdown(message["content"])
    if prompt := st.chat_input("Ask about Greek electricity data / Ρώτησε για ενεργειακά δεδομένα"):
        messages.append({"role": "user", "content": prompt})
        with st.spinner("Retrieving energy data…"):
            result = call_api("POST", "/chat", json={"message": prompt, "context": st.session_state.get("chat_context")})
        if result:
            if result.get("context"):
                st.session_state["chat_context"] = result["context"]
            messages.append({"role": "assistant", "result": result})
            st.rerun()
elif page == "Historical data":
    dataset = st.selectbox("Dataset", ["load", "res", "generation"])
    start = st.date_input("Start date")
    end = st.date_input("End date")
    if st.button("Show history"):
        result = call_api("GET", f"/data/{dataset}", params={"start_date": str(start), "end_date": str(end)})
        if result:
            render(result)
elif page == "Load forecast":
    target = st.date_input("Target date")
    st.caption("Choose one day up to two years after the latest data. Dates more than 14 days ahead use a seasonal projection with historical error bounds and require at least two years of history.")
    if st.button("Forecast load"):
        with st.spinner("Training and forecasting…"):
            result = call_api("GET", "/forecasts/load", params={"date": str(target)})
        if result:
            render(result)
else:
    st.caption("This checks API availability; it does not certify database, model or RAG readiness.")
    if st.button("Check API"):
        try:
            response = requests.get(API_BASE_URL.removesuffix("/api/v1") + "/health", timeout=10)
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException:
            st.error("API unavailable.")
