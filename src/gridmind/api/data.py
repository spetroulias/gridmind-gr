from datetime import date as Date
from typing import Literal
from fastapi import APIRouter, HTTPException
from gridmind.services.analytics import get_analytics

router = APIRouter(prefix="/data", tags=["Historical data"])


@router.get("/{dataset}")
def historical_data(dataset: Literal["load", "res", "generation"],
                    start_date: Date | None = None, end_date: Date | None = None,
                    date: Date | None = None, technology: str | None = None):
    start = start_date or date
    end = end_date or start
    if not start:
        raise HTTPException(422, "Provide date or start_date and end_date.")
    try:
        return get_analytics(dataset, start, end, technology)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Historical data unavailable. Check PostgreSQL and ingest data.") from exc
