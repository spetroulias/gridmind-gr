from fastapi import APIRouter

from gridmind.api.chat import router as chat_router
from gridmind.api.forecasts import router as forecasts_router

api_router = APIRouter(prefix="/api/v1")

# Include individual sub-routers
api_router.include_router(chat_router)
api_router.include_router(forecasts_router)
from gridmind.api.data import router as data_router
api_router.include_router(data_router)
