import os
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel

from app.rag.v2.retriever import get_retriever
from app.models.schemas import PolicyAnswer

load_dotenv()

_retriever = get_retriever(k=5)
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_multi_retriever = MultiQueryRetriever.from_llm(retriever=_retriever, llm=_llm)


class _PolicyList(BaseModel):
    policies: list[PolicyAnswer]


_PERSONA = "당신은 임산부 및 영유아 가정을 위한 복지 정책 안내 전문 챗봇입니다.\n따뜻하고 친근한 말투로 사용자의 상황에 공감하며 답변하세요. 어려운 전문 용어는 쉽게 풀어 설명하고, 사용자가 편안함을 느낄 수 있도록 해주세요."

_RULES = """아래 [참고 문서]를 바탕으로 질문에 답변하세요.
답변 시 다음 규칙을 따르세요:
1. 반드시 policies 배열에 최소 하나 이상의 항목을 반환하세요. 빈 배열은 절대 반환하지 마세요.
2. [참고 문서]에 있는 내용만 사용하고, 문서에 없는 내용은 지어내지 마세요.
3. 복지 정책과 무관한 질문(인사, 잡담 등)이면 policy_name을 "안내"로 하고, content에 챗봇 소개와 함께 어떤 질문을 도와줄 수 있는지 안내하세요.
4. 문서에서 답을 찾을 수 없으면 policy_name을 "안내"로 하고, content에 해당 정보가 없음을 알리고 다른 질문을 유도하세요.
5. 질문과 관련된 복지 정책이 여러 개라면 각 정책을 별도의 항목으로 분리하고, policy_name에는 정책 공식 명칭을 넣으세요.
6. content는 질문에 맞는 내용만 담고, 핵심 정보는 번호나 항목으로 구조화하고 중요한 내용은 강조해서 작성하세요.
7. 사용자가 특정 정책을 이미 신청했거나 완료했다고 알릴 경우, policy_name을 "안내"로 하고, content에 완료 사실을 확인해주며 다른 궁금한 점이 있는지 안내하세요. 해당 정책의 상세 내용을 설명하지 마세요."""

_USER_SECTION = """[기억된 추가 정보] (기초 정보에 없는 추가 맥락 또는 변경사항)
{memory}

[사용자 기초 정보]
{user_info}"""

_DOC_SECTION = """[참고 문서]
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", f"{_PERSONA}\n\n{_RULES}\n\n{_USER_SECTION}\n\n{_DOC_SECTION}"),
        ("human", "{question}"),
    ]
)

REGEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"{_PERSONA}\n\n{_RULES}\n8. [이전 답변]을 참고하여 부족하거나 불명확한 부분을 보완하고, 더 유용하고 풍부한 내용으로 개선된 답변을 제공하세요.\n\n{_USER_SECTION}\n\n[이전 답변]\n{{previous_response}}\n\n{_DOC_SECTION}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


class MemoryExtraction(BaseModel):
    memories: list[str] = []


EXTRACT_MEMORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """대화에서 사용자 본인에 대해 기억할 정보를 추출하세요.

추출 대상: 복지 정책 추천에 유용하지만 [사용자 기초 정보]에 없는 정보
- 이미 신청한 정책 (예: "부모급여 신청 완료")
- 기초 정보의 변경사항 (예: "출산 후로 바뀌었음", "소득 구간 변경")
- 기초 정보에 없는 특수 상황 (예: "남편 실직 상태")

추출 제외:
- 거주지, 임신 여부, 나이, 자녀 수, 임신 주차, 소득 구간 등 기초 정보에 이미 있는 항목
- 제3자(친구, 가족 등) 정보
- 사용자가 명시적으로 언급하지 않은 추측 정보

없으면 빈 리스트 반환. 형식: 간결한 한 문장 (예: "부모급여 신청 완료", "소득 구간 중위 50%로 변경")"""),
    ("human", "사용자 질문: {question}\n\n답변: {answer}"),
])


def extract_memories(question: str, answer: str) -> list[str]:
    try:
        extractor = _llm.with_structured_output(MemoryExtraction)
        result = extractor.invoke(
            EXTRACT_MEMORY_PROMPT.format_messages(question=question, answer=answer)
        )
        return result.memories
    except Exception as e:
        logging.error(f"메모리 추출 오류: {e}", exc_info=True)
        return []


def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", ""))
        header = f"[{source}]" if source else ""
        parts.append(f"{header}\n{doc.page_content}".strip())
    return "\n\n".join(parts)


def get_chain():
    gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = gen_llm.with_structured_output(_PolicyList)
    chain = (
        {
            "context": (lambda x: x["question"]) | _multi_retriever | format_docs,
            "question": lambda x: x["question"],
            "user_info": lambda x: x["user_info"],
            "memory": lambda x: x["memory"],
            "chat_history": lambda x: x["chat_history"],
        }
        | PROMPT
        | structured_llm
    )
    return chain


def get_regenerate_chain():
    gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    structured_llm = gen_llm.with_structured_output(_PolicyList)
    chain = (
        {
            "context": (lambda x: x["question"]) | _multi_retriever | format_docs,
            "question": lambda x: x["question"],
            "user_info": lambda x: x["user_info"],
            "memory": lambda x: x["memory"],
            "previous_response": lambda x: x["previous_response"],
        }
        | REGEN_PROMPT
        | structured_llm
    )
    return chain
