"""Typed analytics dispatch. No model-generated SQL is executed."""
from datetime import date, timedelta
from zoneinfo import ZoneInfo
from datetime import datetime
import json
import re
import pandas as pd
from gridmind.data import queries
from gridmind.rag.retriever_db import DatabaseRetriever, remove_accents


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def parse_request(message):
    text = remove_accents(message)
    today = datetime.now(ZoneInfo("Europe/Athens")).date()
    dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    if dates:
        start = date.fromisoformat(dates[0])
        end = date.fromisoformat(dates[-1])
    elif any(word in text for word in ["in one year", "in a year", "a year from now", "one year later", "σε ενα χρονο"]):
        start = end = (pd.Timestamp(today) + pd.DateOffset(years=1)).date()
    elif any(word in text for word in ["yesterday", "χθες"]):
        start = end = today - timedelta(days=1)
    elif any(word in text for word in ["tomorrow", "αυριο"]):
        start = end = today + timedelta(days=1)
    elif any(word in text for word in ["last month", "περασμενο μηνα", "προηγουμενο μηνα"]):
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
    elif any(word in text for word in ["today", "σημερα"]):
        start = end = today
    else:
        extracted = DatabaseRetriever()._extract_date(message)
        if not extracted:
            return None
        start = end = date.fromisoformat(extracted)
    technology = None
    for candidate, keywords in {
        "lignite": ["lignite", "λιγνιτ"], "natural_gas": ["gas", "αερι"],
        "hydro": ["hydro", "υδρο"], "petroleum": ["petroleum", "oil", "πετρελ"],
    }.items():
        if any(word in text for word in keywords):
            technology = candidate
            break
    if any(word in text for word in ["forecast", "predict", "προβλε", "προγνω"]):
        if any(word in text for word in ["res", "renewable", "απε", "ανανεωσιμ"]):
            raise ValueError("Chat forecasting currently supports system load. Use the RES forecasting CLI for renewables.")
        dataset = "forecast"
    elif technology or any(word in text for word in ["generation", "production", "mix", "source", "παραγωγ", "μιγμα", "πηγ"]):
        dataset = "generation"
    elif any(word in text for word in ["res", "renewable", "απε", "ανανεωσιμ"]):
        dataset = "res"
    else:
        dataset = "load"
    if dataset == "generation" and any(word in text for word in ["renewable", "res", "απε", "ανανεωσιμ"]) and not technology:
        dataset = "res"
    return dataset, start, end, technology


def get_analytics(dataset, start, end, technology=None):
    if start > end:
        raise ValueError("Start date must be on or before end date.")
    if (end - start).days > 366:
        raise ValueError("Please request no more than 367 days at a time.")
    forecast_metadata = None
    if dataset == "load":
        frame = queries.get_historical_load(start, end)
        value, source = "load_mwh", "RealTimeSCADASystemLoad"
    elif dataset == "res":
        frame = queries.get_historical_res(start, end)
        value, source = "res_mwh", "RealTimeSCADARES"
    elif dataset == "generation":
        frame = queries.get_hourly_generation_mix(start, end)
        if technology:
            queries._validate_technology(technology)
            frame = frame[frame["technology"] == technology]
        value, source = "production_mwh", "SystemRealizationSCADA"
    elif dataset == "forecast":
        from gridmind.models import load_forecast as model
        if start != end:
            raise ValueError("Choose one target date for a load forecast.")
        history = model.load_history()
        latest = history["timestamp"].max().normalize()
        target = pd.Timestamp(start)
        if target <= latest:
            raise ValueError("Choose a forecast date after the latest observations; historical simulations are available through the CLI.")
        if target > latest + pd.Timedelta(days=14):
            from gridmind.models.long_term_load import make_long_term_forecast
            frame, forecast_metadata = make_long_term_forecast(history, target)
            source = "GridMind calendar-based seasonal load projection trained on RealTimeSCADASystemLoad"
        else:
            fitted = model.train_model(history)
            frame = model.make_future_forecast(fitted, history, target)
            source = "GridMind Random Forest trained on RealTimeSCADASystemLoad"
        value = "forecast_mwh"
    else:
        raise ValueError("Unknown dataset.")
    if frame.empty:
        return {"answer": f"No {dataset} data found for {start} to {end}.", "sources": [source], "data": [], "dataset": dataset}
    # Generation summaries refer to the aggregate system/source, not individual units.
    series = frame.groupby("timestamp")[value].sum() if dataset == "generation" else frame.set_index("timestamp")[value]
    peak = series.idxmax()
    answer = (f"{dataset.title()}, {start} to {end}: {len(series)} hourly {'estimates' if dataset == 'forecast' else 'observations'}. "
              f"Total: {series.sum():,.2f} MWh; mean: {series.mean():,.2f} MWh per hour; "
              f"peak: {series.max():,.2f} MWh at {peak}; "
              f"minimum: {series.min():,.2f} MWh at {series.idxmin()}.")
    if dataset == "load":
        answer += " Load means net system load without Crete, not total Greek electricity consumption."
    answer += " Uses reporting periods 1–24 as a nominal hourly view; daylight-saving days may have incomplete totals or shifted clock labels."
    if dataset == "forecast":
        answer += " These are model estimates, not measured consumption or official ADMIE forecasts."
    if forecast_metadata:
        answer += (
            " Long-term seasonal projection assuming historical calendar patterns persist. "
            "Future weather, holidays and structural demand changes are not modeled. "
            f"Held-out year MAE: {forecast_metadata['validation_metrics']['mae_mwh']:,.2f} MWh; "
            f"WAPE: {forecast_metadata['validation_metrics']['wape_pct']:.2f}%. "
            "The chart bounds use the 90th percentile of historical absolute errors; "
            "they are not a guarantee of future coverage."
        )
    return {"answer": answer, "sources": [source], "data": records(frame), "dataset": dataset,
            "forecast_metadata": forecast_metadata}
