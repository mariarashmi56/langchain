from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
import os

load_dotenv()

if __name__ == '__main__':
    print("Ingesting.....")
    loader = TextLoader("/workspaces/langchain/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()

    print("Splitting.....")
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks of text")

    embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("OPENAI_API_KEY"))
    print("Creating Pinecone index.....")
    PineconeVectorStore.from_documents(texts, embedding=embeddings, index_name =os.environ['INDEX_NAME'])
    print("Ingestion complete.")
    