import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

PDF_DIR = "data/pdfs/2026318_PDF"
FAISS_DIR = "faiss_index"


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_index():
    docs = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"PDF {len(pdf_files)}개 로드 중...")
    for fname in pdf_files:
        loader = PyPDFLoader(os.path.join(PDF_DIR, fname))
        docs.extend(loader.load())
    print(f"총 {len(docs)}페이지 로드 완료")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"청크 수: {len(chunks)}")

    print("벡터DB 구축 중...")
    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"faiss_index/ 저장 완료")

    return vectorstore


def load_index():
    return FAISS.load_local(FAISS_DIR, get_embeddings(), allow_dangerous_deserialization=True)


if __name__ == "__main__":
    vectorstore = build_index()
    results = vectorstore.similarity_search("revenue", k=3)
    print(f"\n[similarity_search('revenue')] {len(results)}개 청크 반환")
    for i, r in enumerate(results):
        print(f"  chunk {i+1} (page {r.metadata.get('page')}, source {os.path.basename(r.metadata.get('source', ''))})")
        print(f"  {r.page_content[:100].encode('ascii', 'ignore').decode()}...")
