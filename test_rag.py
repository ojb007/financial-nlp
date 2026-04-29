from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# 1. PDF 로드
print("PDF 로드 중...")
loader = PyPDFLoader("data/pdfs/2026318_PDF/AAPL-10K-2025.pdf")
docs = loader.load()
print(f"페이지 수: {len(docs)}")

# 2. 청크 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"청크 수: {len(chunks)}")

# 3. 임베딩 + FAISS
print("벡터화 중...")
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 4. 프롬프트 + 체인
prompt = ChatPromptTemplate.from_template("""Answer based on the context below.

Context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 질문
question = "What were Apple's net sales for fiscal year 2024?"
print(f"\nQ: {question}")
answer = chain.invoke(question)
print(f"A: {answer}")
