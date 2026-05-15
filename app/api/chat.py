from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/ai/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request):
    user_info_str = ""
    if request.user_info:
        data = request.user_info.model_dump(exclude_none=True)
        user_info_str = "\n".join(f"- {k}: {v}" for k, v in data.items())

    answer = req.app.state.chain.invoke({
        "question": request.user_message,
        "user_info": user_info_str,
    })
    return ChatResponse(answer=answer)
