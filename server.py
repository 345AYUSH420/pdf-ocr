import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ocr_loader import load_pdf_with_ocr
from vector_store import setup_vector_store

load_dotenv()

# Load scanned PDF with OCR
docs = load_pdf_with_ocr("00000001.tif.pdf")

# Setup vector DB
retriever = setup_vector_store(docs)

# LLM on OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# Strict RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict RAG assistant. Answer ONLY using the provided context. "
     "If the answer is not present in the context, reply exactly: 'Not found in the provided text.'"),
    ("human",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Answer strictly using the context.")
])

parser = StrOutputParser()

# RAG Chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# Query
query = "इस संस्करण के पुस्तक के प्रकाशक कौन हैं?"
response = rag_chain.invoke(query)

print("\n🟩 Final Answer:\n", response)
