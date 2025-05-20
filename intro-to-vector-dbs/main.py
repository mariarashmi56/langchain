from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain import hub
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


if __name__ == "__main__":
    # Initialize Pinecone
    print('Retrieving ...')
    prompt = "What is a vector store - why is it needed?"

    llm = ChatOpenAI()
    # chain = PromptTemplate(template=prompt) | llm
    # res = chain.invoke(input={})
    #print(res.content)
    embedding = OpenAIEmbeddings()
    
    vector_store= PineconeVectorStore(index_name = os.getenv("INDEX_NAME"),embedding=embedding)
    qa_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    create_documents = create_stuff_documents_chain(llm = llm, prompt = qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=create_documents)
    result = retrieval_chain.invoke(input={"input": prompt})
    print(result)
# The above code is a simple example of how to use LangChain with OpenAI's ChatGPT model to generate a response to a prompt about Pinecone in machine learning.



