from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, PolicyAnswer, RAGResponse, UserInfo
from app.rag.v2.chain import extract_memories

router = APIRouter()

# 사용자 정보 필드에 대한 한글 레이블 매핑
_FIELD_LABELS = {
    "district": "거주 자치구",
    "pregnancy_status": "임신 여부",
    "age": "만 나이",
    "children_count": "자녀 수",
    "pregnancy_weeks": "임신 주차",
    "child_age_months": "자녀 생후 개월 수",
    "multiple_birth": "다태아",
    "income_level": "소득 구간",
    "is_korean": "국적",
    "no_house": "무주택",
}


def _format_user_info(user_info: UserInfo) -> str:
    lines = []
    data = user_info.model_dump(exclude_none=True)
    for field, value in data.items():
        label = _FIELD_LABELS.get(field, field)
        if field == "is_korean":
            value = "내국인" if value else "외국인"
        elif isinstance(value, bool):
            value = "해당" if value else "해당 없음"
        elif hasattr(value, "value"):
            value = value.value
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


@router.post("/ai/chat", response_model=RAGResponse)
def chat(request: ChatRequest, req: Request):
    user_info_str = _format_user_info(request.user_info) if request.user_info else "없음"
    memory_str = "\n".join(f"- {m}" for m in request.memory) if request.memory else "없음"

    chain = req.app.state.chain_v2_regenerate if request.regenerate else req.app.state.chain_v2
    invoke_input = {
        "question": request.user_message,
        "user_info": user_info_str,
        "memory": memory_str,
    }
    if request.regenerate:
        invoke_input["previous_response"] = request.previous_response or "없음"

    result = chain.invoke(invoke_input)

    answer_text = "\n".join(p.content for p in result.policies)
    new_memories = [] if request.regenerate else extract_memories(request.user_message, answer_text)

    policies = [
        PolicyAnswer(
            policy_name=None if p.policy_name == "안내" else p.policy_name,
            content=p.content,
        )
        for p in result.policies
    ]
    return RAGResponse(policies=policies, new_memories=new_memories)
