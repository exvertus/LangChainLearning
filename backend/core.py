import os
from typing import Any
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.pinecone import Pinecone
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT"])

def run_llm(query:str) -> Any:
    embedding = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=os.environ["PINECONE_INDEX"],
        embedding=embedding)
    chat = ChatOpenAI(temperature=0, verbose=True)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )
    return qa({"query": query})

if __name__ == "__main__":
    print(run_llm("What is langchain?"))
