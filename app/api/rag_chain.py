from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

FAISS_DIR = "faiss_index"

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    return _vectorstore


def build_chain(model: str = "gemini-2.0-flash", k: int = 5):
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template("""You are a financial analyst. Answer the question based on the context below.

Context:
{context}

Question: {question}
""")

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def invoke(question: str, model: str = "gemini-2.0-flash", k: int = 5) -> str:
    chain = build_chain(model=model, k=k)
    return chain.invoke(question)
