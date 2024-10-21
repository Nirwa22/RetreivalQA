from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()
Hugging_face_Api = os.getenv("Huggingface_Api_Key")


def vectordb(docs):
    document = PyPDFDirectoryLoader(docs)
    all_documents = document.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                              chunk_overlap=20,
                                              length_function=len,
                                              is_separator_regex=False)
    chunks = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_database = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="Vectordb"
    )
    return vector_database