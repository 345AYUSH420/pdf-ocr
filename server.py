import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from vision_ocr_loader import load_pdf_with_vision_ocr, read_ocr_output
from langchain_core.documents import Document

from ocr_loader import load_pdf_with_ocr
from vector_store import setup_vector_store

load_dotenv()


# load_pdf_with_vision_ocr(
#     input_gcs_uri="gs://ocr-extraction-bucket/00000001.tif.pdf" ,
#     output_gcs_uri= "gs://ocr-extracted-bucket/output/")

ocr_text = read_ocr_output("gs://ocr-extracted-bucket/output/")

docs = [Document(page_content=ocr_text)]
# print("ocr_text" , type(ocr_text))

# print("Extracted OCR Text Length:", len(ocr_text))

# Setup vector DB
retriever = setup_vector_store(docs)

# LLM on OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    max_tokens=512,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# Strict RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a multilingual RAG assistant that MUST translate answers to match the user's question language. "
     "Step 1: Use the provided context to find the answer. "
     "Step 2: Translate the answer to the user's language. "
     "Rules: "
     "- If the question is in English, translate your answer to English. "
     "- If the question is in Hindi, answer in Hindi. "
     "- Always provide the answer in the user's language, NOT the source language. "
     "If the answer is not in context, reply: 'Not found in the provided text.'"),
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

while True:
    query = input("\n🟦 Enter your question about the document (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = rag_chain.invoke(query)

    print("\n🟩 Final Answer:\n", response)
