from fastapi import APIRouter
from app.api.v1.endpoints import groq

api_router = APIRouter()

# Include the groq router with a prefix
api_router.include_router(groq.router, prefix="/groq", tags=["Groq"])
