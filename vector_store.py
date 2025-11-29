from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def setup_vector_store(docs):

    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY")
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "books"


    print("index name:" , index_name)


    index_exists = index_name in [i["name"] for i in pc.list_indexes()]
    print("Index exists:", index_exists)


    if not index_exists:

        print("➡️ Creating new index & uploading embeddings (first run)...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name, 
            namespace="default"
        )
    else:
        print("➡️ Index exists — reusing existing vectors (no embedding).")
        vectorstore = PineconeVectorStore.from_existing_index(
            embedding=embeddings,
            index_name=index_name,
            namespace="default"
        )

    return vectorstore.as_retriever(search_kwargs={"k": 5})
