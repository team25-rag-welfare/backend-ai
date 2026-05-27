import os
import logging
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel

from app.rag.v2.retriever import get_retriever
from app.models.schemas import PolicyAnswer

load_dotenv()

_retriever = get_retriever(k=3)
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_multi_retriever = MultiQueryRetriever.from_llm(retriever=_retriever, llm=_llm)


class _PolicyList(BaseModel):
    policies: list[PolicyAnswer]


_PERSONA = """당신은 임산부 및 영유아 가정을 위한 복지 정책 안내 전문 챗봇입니다.
사용자가 복지 제도를 어렵게 느낄 수 있다는 점을 이해하고, 친절하고 부드러운 말투로 안내하세요.
답변은 정확하되 딱딱한 행정 문서처럼 쓰지 말고, 보호자에게 설명하듯 쉽게 풀어 말하세요.
사용자가 바로 이해하고 다음 행동을 할 수 있도록 핵심을 정리해 주세요."""

_RULES = """아래 [참고 문서]를 바탕으로 질문에 답변하세요.
답변 시 다음 규칙을 따르세요:
1. 반드시 policies 배열에 최소 하나 이상의 항목을 반환하세요. 빈 배열은 절대 반환하지 마세요.
2. [참고 문서]에 있는 내용만 사용하고, 문서에 없는 내용은 지어내지 마세요.
3. 복지 정책과 무관한 질문(인사, 잡담 등)이면 policy_name을 "안내"로 하고, content에 챗봇 소개와 함께 어떤 질문을 도와줄 수 있는지 안내하세요.
4. 문서에서 답을 찾을 수 없으면 policy_name을 "안내"로 하고, content에 해당 정보가 없음을 알리고 다른 질문을 유도하세요.
5. 질문과 관련된 복지 정책이 여러 개라면 각 정책을 별도의 항목으로 분리하고, policy_name에는 정책 공식 명칭을 넣으세요.
6. content는 질문에 맞는 내용만 담고, 핵심 정보는 번호나 항목으로 구조화하고 중요한 내용은 강조해서 작성하세요.
7. 답변은 바로 정보만 나열하지 말고, 사용자의 질문을 짧게 받아주는 문장으로 시작하세요.
8. 정책 정보는 항목형으로 정리하되, 각 항목은 사용자가 이해하기 쉬운 말투로 설명하세요. 행정 문서처럼 딱딱한 표현은 피하세요.
9. 답변 마지막에는 사용자가 이어서 물어볼 수 있는 관련 질문을 자연스럽게 제안하세요.
10. 여러 정책이 있는 경우, 첫 번째 정책 답변에서만 짧은 시작 문장을 넣고 마지막 정책 답변에서만 마무리 질문을 넣으세요. 각 정책마다 같은 인사말이나 마무리 문장을 반복하지 마세요.
11. 단, 친절하게 보이기 위해 [참고 문서]에 없는 내용을 추측하지 마세요.
12. 사용자가 특정 정책을 이미 신청했거나 완료했다고 알릴 경우, policy_name을 "안내"로 하고, content에 완료 사실을 확인해주며 다른 궁금한 점이 있는지 안내하세요. 해당 정책의 상세 내용을 설명하지 마세요."""

_USER_SECTION = """[기억된 추가 정보] (기초 정보에 없는 추가 맥락 또는 변경사항)
{memory}

[사용자 기초 정보]
{user_info}"""

_DOC_SECTION = """[참고 문서]
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", f"{_PERSONA}\n\n{_RULES}\n\n{_USER_SECTION}\n\n{_DOC_SECTION}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

REGEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{_PERSONA}

{_RULES}

재생성 추가 규칙:
- 지금 요청은 '응답 재생성'입니다. 사용자는 이전 답변이 부족하다고 느껴 더 나은 답변을 원합니다.

- [이전 답변]은 참고만 하세요. 이전 답변과 같은 문장, 같은 순서, 같은 구성으로 반복하지 마세요.

- 이전 답변을 단순히 말만 바꾸어 다시 쓰지 마세요. 반드시 새로운 설명 흐름으로 다시 작성하세요.

- 이전 답변과 같은 결론이더라도, 설명 방식은 반드시 다르게 구성하세요. 이전 답변이 짧았다면 사용자가 오해하기 쉬운 점, 조건, 예외, 신청 시점에 따른 차이, 한 문장 요약 중 [참고 문서]에서 확인되는 내용을 보완하세요.

- 정책 질문이라면 사용자가 바로 이해할 수 있도록 가능한 한 아래 항목을 중심으로 구체적으로 답변하세요:
- 지원 대상
- 지원 금액 또는 혜택 내용
- 신청 방법
- 사용처 또는 지급 방식
- 신청/사용 기한
- 사용자의 상황과 관련된 조건
- 주의사항
[이전 답변]에 이미 포함된 내용은 짧게 정리하고, [참고 문서]에서 새로 확인되는 신청 방법, 신청 기관, 지급일, 처리기한, 변경신청, 주의사항을 우선 보완하세요.

재생성 답변은 이전 답변보다 정보량이 늘어나야 합니다. 단순히 금액만 다시 반복하지 말고, 사용자가 다음 행동을 할 수 있도록 신청/지급/주의사항 정보를 추가하세요.

- "소급 지원", "신청 기간", "대상 여부"처럼 조건을 묻는 짧은 후속 질문에는 결론만 말하지 말고, 기준 시점과 사용자가 주의할 점을 함께 설명하세요.

- [참고 문서]에 금액, 기간, 대상 조건, 신청 방법, 사용처가 있다면 생략하지 말고 답변에 포함하세요.

- [참고 문서]에서 확인되지 않는 정보는 추측하지 마세요. 확인되지 않는 항목은 억지로 채우지 말고 자연스럽게 생략하거나, 필요한 경우 "제공된 문서에서는 확인되지 않습니다"라고 안내하세요.

- 여러 정책이 관련되어 있다면 정책별로 나누어 답변하세요. 각 정책은 사용자가 비교하기 쉽도록 핵심 정보를 항목형으로 정리하세요.

- 복지 정책과 무관한 질문이라면 이전 안내 문구를 반복하지 말고, 자연스럽고 짧게 다시 답변하세요. 단, 챗봇의 역할은 임산부 및 영유아 가정을 위한 복지 정책 안내임을 알려주세요.

{_USER_SECTION}

[이전 답변]
{{previous_response}}

{_DOC_SECTION}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
print("저장된 메모리다. : {memory}")


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
- 이미 저장된 메모리

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


def limit_docs(docs, max_docs: int = 5):
    return docs[:max_docs]


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text or "").lower()


def _filter_docs_by_policy(docs, current_policy: str):
    if not current_policy or current_policy == "없음":
        return docs

    normalized_policy = _normalize_for_match(current_policy)
    if not normalized_policy:
        return docs

    matched = [
        doc for doc in docs
        if normalized_policy in _normalize_for_match(doc.page_content)
    ]
    return matched if matched else docs


def _retrieve_context(query: str, current_policy: str) -> str:
    docs = _multi_retriever.invoke(query)
    docs = _filter_docs_by_policy(docs, current_policy)
    docs = limit_docs(docs)
    return format_docs(docs)


def _build_search_query(question: str, chat_history, current_policy: str) -> str:
    recent_messages = chat_history[-2:] if chat_history else []
    recent = "\n".join(
        f"{msg.type}: {msg.content}" for msg in recent_messages
    )
    return f"""현재 이어지는 정책:
{current_policy}

최근 대화:
{recent if recent else "없음"}

사용자 질문:
{question}

현재 이어지는 정책이 있으면 그 정책을 가장 우선해서 검색하세요.
현재 이어지는 정책이 없으면 최근 대화에서 이어지는 정책명이나 주제를 기준으로 검색하세요.
정책의 지원 대상, 자격 확인 방법, 소득 기준, 신청 방법을 중심으로 검색하세요."""


def get_chain():
    gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = gen_llm.with_structured_output(_PolicyList)
    chain = (
        {
            "context": lambda x: _retrieve_context(
                _build_search_query(
                    x["question"],
                    x["chat_history"],
                    x["current_policy"],
                ),
                x["current_policy"],
            ),
            "question": lambda x: x["question"],
            "user_info": lambda x: x["user_info"],
            "memory": lambda x: x["memory"],
            "chat_history": lambda x: x["chat_history"],
            "current_policy": lambda x: x["current_policy"],
        }
        | PROMPT
        | structured_llm
    )
    return chain


def _build_regenerate_query(question: str, previous_response: str, current_policy: str) -> str:
    return f"""{question}

현재 이어지는 정책:
{current_policy}

이전 답변:
{previous_response}

현재 이어지는 정책이 있으면 그 정책을 가장 우선해서 검색하세요.
현재 이어지는 정책이 없으면 위 질문과 이전 답변에 나온 정책의 핵심 정보를 같은 의미의 다른 표현까지 포함해 다시 검색하세요.
검색 초점은 지원 금액, 지원 대상, 지급 방식, 신청 방법입니다.
보호자 변경, 미지급 급여, 환수, 과태료, 입양, 사망, 행정 처리 절차 등 특수 상황은 사용자가 직접 묻지 않았다면 제외하세요."""

def get_regenerate_chain():
    gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    structured_llm = gen_llm.with_structured_output(_PolicyList)
    chain = (
        {
            "context": lambda x: _retrieve_context(
                _build_regenerate_query(
                    x["question"],
                    x["previous_response"],
                    x["current_policy"],
                ),
                x["current_policy"],
            ),
            "question": lambda x: x["question"],
            "user_info": lambda x: x["user_info"],
            "memory": lambda x: x["memory"],
            "previous_response": lambda x: x["previous_response"],
            "chat_history": lambda x: x["chat_history"],
            "current_policy": lambda x: x["current_policy"],
        }
        | REGEN_PROMPT
        | structured_llm
    )
    return chain
