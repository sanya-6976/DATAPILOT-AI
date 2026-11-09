from fastapi import FastAPI

from backend.app.api.routes.datasets import router as datasets_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "An autonomous, evidence-driven "
        "data research and intelligence platform."
    ),
    version="0.1.0",
)

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health"],
)

app.include_router(
    datasets_router,
    prefix="/api/v1/datasets",
    tags=["Datasets"],
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to DataPilot AI",
        "status": "running",
        "version": "0.1.0",
    }
