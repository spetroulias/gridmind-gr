import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

# Ορίζουμε το router τοπικά (ΔΕΝ κάνουμε import από το forecasts!)
router = APIRouter(prefix="/forecasts", tags=["Forecasts"])


@router.get("/", response_model=Dict[str, Any])
async def get_forecast_status() -> Dict[str, Any]:
    """Endpoint για επιβεβαίωση της λειτουργίας των ενεργειακών προβλέψεων."""
    try:
        logger.info("Fetching energy forecast status...")
        return {
            "status": "success",
            "message": "GridMind Forecasts API is active.",
            "available_models": ["load_forecast", "res_forecast"],
        }
    except Exception as e:
        logger.error(f"Error fetching forecast status: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve forecast data."
        )