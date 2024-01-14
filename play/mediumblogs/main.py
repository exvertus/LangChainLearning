import os
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
import pinecone

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment='gcp-starter')

if __name__ == '__main__':
    loader = TextLoader(str(Path(__file__).parent / 'mediumblog1.txt'))
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_documents(texts, embeddings, index_name='langchain-doc-index')

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
