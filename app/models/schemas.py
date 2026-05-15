from pydantic import BaseModel
from typing import Any, List, Optional

class UserInfo(BaseModel):
    age: Optional[int] = None
    district: Optional[str] = None
    pregnancy_weeks: Optional[str] = None

class ChatRequest(BaseModel):
    user_info: Optional[UserInfo] = None
    memory: List[str] = []
    user_message: str

class ChatResponse(BaseModel):
    answer: str