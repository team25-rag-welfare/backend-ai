from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.rag.v1.retriever import get_retriever

load_dotenv()

_retriever = get_retriever(k=3)
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 임산부 및 영유아 가정을 위한 복지 정책 안내 전문 챗봇입니다.
아래 제공된 문서를 참고하여 질문에 답변하세요.
문서에 없는 내용은 "해당 정보를 찾을 수 없습니다."라고 답변하세요.
답변은 한국어로 친절하게 작성하세요.

[사용자 정보]
{user_info}

[참고 문서]
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain():
    chain = (
        {
            "context": (lambda x: x["question"]) | _retriever | format_docs,
            "question": lambda x: x["question"],
            "user_info": lambda x: x["user_info"],
        }
        | PROMPT
        | _llm
        | StrOutputParser()
    )
    return chain
