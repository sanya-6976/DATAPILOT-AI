from fastapi import APIRouter

from backend.app.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_settings()

    return {
        "status": "healthy",
        "application": settings.app_name,
        "environment": settings.app_env,
        "version": "0.1.0",
    }
