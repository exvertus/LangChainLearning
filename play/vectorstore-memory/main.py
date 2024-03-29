import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == '__main__':
    pdf_path = Path(__file__).parent / 'reactlm.pdf'
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local('reactlm_index')
    
    new_vectorstore = FAISS.load_local('reactlm_index', embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)
