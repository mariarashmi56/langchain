from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():

    loader = ReadTheDocsLoader('/workspaces/langchain/documentation-helper/langchain-docs')
    raw_documents = loader.load()
    print(f"{len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata['source']
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    print(f"Loading {len(documents)} into Pinecone Vector Store")

    PineconeVectorStore.from_documents(documents=documents, embedding=embeddings,
                                       index_name = "langchain-doc-index"
    )

if __name__ == "__main__":
    print("Starting ingestion...")
    ingest_docs()
    print("Ingestion complete.")





