import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# 경로 설정
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pdf")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")


def load_pdfs(pdf_dir: str):
    """data/pdf/ 폴더 안의 모든 PDF 파일을 페이지 단위로 로드"""
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_dir}")

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        documents.extend(docs)
        print(f"  - {filename}: {len(docs)}페이지 로드")

    return documents


def split_documents(documents):
    """문서를 청크 단위로 분할 (chunk_size=1000, chunk_overlap=100)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


def save_to_chroma(chunks, persist_dir: str):
    """청크를 임베딩하여 ChromaDB에 저장 (기존 DB 초기화 후 재저장)"""
    # 기존 DB가 있으면 삭제 후 재생성 (중복 방지)
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("  - 기존 ChromaDB 초기화 완료")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vector_db


def ingest():
    print("=== PDF 데이터 적재 시작 ===")

    print(f"\n[1/3] PDF 로드 중... ({PDF_DIR})")
    documents = load_pdfs(PDF_DIR)
    print(f"-> 총 {len(documents)}페이지 로드 완료")

    print("\n[2/3] 텍스트 청킹 중...")
    chunks = split_documents(documents)
    print(f"-> 총 {len(chunks)}개 청크 생성 완료")

    print("\n[3/3] 임베딩 및 ChromaDB 저장 중... (OpenAI API 호출 발생)")
    save_to_chroma(chunks, CHROMA_DIR)
    print(f"-> ChromaDB 저장 완료: {CHROMA_DIR}")

    print("\n=== 적재 완료 ===")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        print(f"\n[오류] {e}")
