from pydantic import BaseModel
from typing import List, Optional


# 사용자 정보
class UserInfo(BaseModel):
    # 필수
    district: str
    pregnancy_status: str
    age: int
    children_count: int

    # 선택
    pregnancy_weeks: Optional[int] = None
    child_age_months: Optional[int] = None
    multiple_birth: Optional[bool] = None
    income_level: Optional[int] = None
    is_korean: Optional[bool] = None
    no_house: Optional[bool] = None


# 사용자 질문
class ChatRequest(BaseModel):
    user_info: Optional[UserInfo] = None
    memory: List[str] = []
    user_message: str
    recent_chats : List[dict[str, str]] = []
    #재생성 여부
    regenerate: bool = False
    previous_response: Optional[str] = None


# AI 답변 (v1)
class ChatResponse(BaseModel):
    answer: str


# AI 답변 (v2)
class PolicyAnswer(BaseModel):
    policy_name: Optional[str]
    content: str

# 메모리 추출
class RAGResponse(BaseModel):
    policies: list[PolicyAnswer]
    new_memories: list[str] = []
