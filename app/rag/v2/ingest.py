import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "pdf")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "chroma_db_v2")


def load_pdfs(pdf_dir: str):
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_dir}")
    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        full_text = "\n".join(page.page_content for page in pages)
        documents.append(Document(page_content=full_text, metadata={"source": filepath}))
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def save_to_chroma(chunks, persist_dir: str):
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vector_db


def ingest():
    documents = load_pdfs(PDF_DIR)
    chunks = split_documents(documents)
    save_to_chroma(chunks, CHROMA_DIR)
    print(f"완료: {len(chunks)}개 청크 저장됨 → {CHROMA_DIR}")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        print(f"\n[오류] {e}")
