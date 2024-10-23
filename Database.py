from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
    vector_database = FAISS.from_documents(
        chunks,
        embeddings
    )
    return vector_database


input_file_path = "Pdf_files"
v_database = vectordb(input_file_path)
v_database.save_local("Vectordb")
print("done")