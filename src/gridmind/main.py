from fastapi import FastAPI
from gridmind.api.router import api_router

app = FastAPI(
    title="GridMind GR API",
    version="1.0.0",
    description="Unified API combining Structured Energy Forecasts & Unstructured ADMIE RAG Pipeline",
)

# Mount central API router containing /chat and /forecasts
app.include_router(api_router)

@app.get("/")
def health_check():
    return {"status": "online", "system": "GridMind GR API"}