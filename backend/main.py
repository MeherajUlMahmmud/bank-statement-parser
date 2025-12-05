from app.api.v1.api import api_router
from fastapi import FastAPI
from app.core.config import settings
import logging
from logging.config import dictConfig

from app.core.logging import LOGGING_CONFIG

# 1. Apply the logging configuration BEFORE importing FastAPI
# This ensures all loggers are configured correctly from the start.
dictConfig(LOGGING_CONFIG)

# Get a logger instance for this module
logger = logging.getLogger(__name__)


app = FastAPI(
    title="GroqCloud Proxy API",
    description="A structured FastAPI application to proxy requests to the GroqCloud Llama models.",
    version="1.0.0",
)

# 2. Add a startup event to confirm logging is working


@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

# Include the main API router
app.include_router(api_router, prefix="/api/v1")

# 3. Add a simple root endpoint that uses the logger


@app.get("/")
async def read_root():
    logger.info("Root endpoint was accessed.")
    return {"message": "Welcome to the Awesome API! Check the console and logs/ directory."}
