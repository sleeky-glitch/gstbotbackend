# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    # Removed namespace parameter since we'll query all namespaces

class Reference(BaseModel):
    file_name: str
    page: int
    namespace: str  # Added namespace to show which document it came from

class QueryResponse(BaseModel):
    response: str
    success: bool
    references: List[Reference]
