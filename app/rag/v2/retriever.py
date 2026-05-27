import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "chroma_db_v2")


def get_retriever(k: int = 5):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        }
    )

    return retriever
