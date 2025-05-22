# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class Reference(BaseModel):
    file_name: str
    page: int
    namespace: str
    index_name: str  # Added index name to show which index it came from

class QueryResponse(BaseModel):
    response: str
    success: bool
    references: List[Reference]
