from fastapi import APIRouter

from .endpoints import statements, export

api_router = APIRouter()

api_router.include_router(statements.router, prefix="/statements", tags=["statements"])
api_router.include_router(export.router, prefix="/statements", tags=["export"])
