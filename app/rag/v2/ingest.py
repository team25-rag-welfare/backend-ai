import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

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
        loader = PyMuPDFLoader(filepath)
        pages = loader.load()
        documents.extend(pages)
        print(f"  - {filename}: {len(pages)}페이지 로드")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
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
