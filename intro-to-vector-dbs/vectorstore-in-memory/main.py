from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


if __name__ == "__main__":
   
   llm = ChatOpenAI()

   path = "/workspaces/langchain/2210.03629v3.pdf"
   loader = PyPDFLoader(file_path=path)
   documents = loader.load()
   text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=30, separator="\n")
   docs = text_splitter.split_documents(documents = documents)
   
    # Create embeddings
   embeddings = OpenAIEmbeddings()
   vectorstore = FAISS.from_documents(docs, embeddings)
   vectorstore.save_local("faiss_index_react")

   new_vector_store = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
   qa_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
   create_documents = create_stuff_documents_chain(llm = llm, prompt = qa_prompt)
   retrieval_chain = create_retrieval_chain(retriever=new_vector_store.as_retriever(), combine_docs_chain=create_documents)
   result = retrieval_chain.invoke(input={"input": "What is React , explain it to me in three sentences"})
   print(result['answer'])

