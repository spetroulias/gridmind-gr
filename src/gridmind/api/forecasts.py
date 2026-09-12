from datetime import date
from fastapi import APIRouter, HTTPException
from gridmind.services.analytics import get_analytics

router = APIRouter(prefix="/forecasts", tags=["Forecasts"])


@router.get("/")
def forecast_status():
    return {"status": "available", "endpoint": "/api/v1/forecasts/load?date=YYYY-MM-DD"}


@router.get("/load")
def load_forecast(date: date):
    try:
        return get_analytics("forecast", date, date)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Forecast unavailable. Check PostgreSQL and historical coverage.") from exc
