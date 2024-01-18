import os
from typing import Any, List, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.pinecone import Pinecone
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT"])

def run_llm(query:str, chat_history: List[Dict[str,Any]]) -> Any:
    embedding = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=os.environ["PINECONE_INDEX"],
        embedding=embedding)
    chat = ChatOpenAI(temperature=0, verbose=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})

if __name__ == "__main__":
    print(run_llm("What is langchain?"))
