"""FastAPI application factory for NetraRakshakAI."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.inference.predictor import ScreeningPredictor

logger = logging.getLogger("netrarakshak_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifecycle of the frozen E015 inference predictor singleton."""
    logger.info("Initializing E015 ScreeningPredictor singleton...")
    try:
        app.state.predictor = ScreeningPredictor()
        logger.info("ScreeningPredictor loaded on %s", app.state.predictor.device)
    except Exception as e:
        logger.error("Failed to load ScreeningPredictor: %s", e)
        app.state.predictor = None
    yield
    logger.info("Shutting down NetraRakshakAI API server.")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title="NetraRakshakAI Screening API",
        version="1.0.0-phase16a",
        description="FastAPI interface exposing the frozen E015 diabetic retinopathy screening pipeline.",
        lifespan=lifespan,
    )

    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
