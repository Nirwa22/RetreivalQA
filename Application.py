from Database import v_database
from flask import Flask, request
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
Application = Flask(__name__)
CORS(Application)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
new_vector_store = FAISS.load_local(
    "Vectordb", embeddings, allow_dangerous_deserialization=True
)

@Application.route("/")
def homepage():
    return "Home Route"


@Application.route("/Upload_files", methods=['POST'])
def upload_files():
    input_file_path = "Pdf_files"
    return v_database


@Application.route("/Enter_Query", methods=['POST'])
def enter_query():
    input = request.get_json()
    retriever = new_vector_store.as_retriever
    return retriever.invoke(input["question"])


if "__main__" == __name__:
    Application.run(debug=True)