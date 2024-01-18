import os
import re
from pathlib import Path
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT"])

def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path=str(Path(__file__).parent / 
                                        'langchain-docs/.langchain-docs/langchain.readthedocs.io/en/latest'),
                                        encoding='utf-8')
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=['\n\n', '\n', ' ', '']
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")

    for doc in documents:
        del doc.metadata["source"]

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name="langchain-docs")
    print("Added to Pinecone vectorstore")

if __name__ == "__main__":
    ingest_docs()
