from typing import Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    role: Literal["system", "user", "assistant"]
    message: str = Field()
