from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from flask import Flask, request
from flask_cors import CORS
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")
api_key = os.getenv("Api_Token")

Application = Flask(__name__)
CORS(Application)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
new_vector_store = FAISS.load_local("Vectordb", embeddings, allow_dangerous_deserialization=True)


def output_response(query):
    template_sample = """
    Answer the following based on the following:
    {context}
    Question:{input}
    """
    # results = new_vector_store.similarity_search_with_relevance_scores(query, k=1)
    # context_text = list(results[0])[0].page_content
    final_prompt = ChatPromptTemplate.from_template(template_sample)
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                     temperature=0,
                     max_tokens=None
                     )
    retriever = new_vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 1})
    # retriever2 = new_vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1, "k": 1})
    chain = create_stuff_documents_chain(llm, prompt=final_prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]


@Application.route("/")
def homepage():
    return "Home Route"


@Application.route("/Upload_files", methods=['POST'])
def upload_files():
    input_file_path = "Pdf_files"
    return f"The vector database for {input_file_path} is created and stored at the path: Vectordb"


@Application.route("/Enter_Query", methods=['POST'])
def enter_query():
    api = request.headers.get("Authorization")
    if api == api_key:
        try:
            input = request.get_json()
            if input["Search"]:
                return output_response(input["Search"])
            else:
                return {"Message": "Enter your query"}
        except Exception as e:
            return e
    elif api and api != api_key:
        return{"Message": "Unauthorized access"}
    else:
        return{"Message": "Api key needed"}


if "__main__" == __name__:
    Application.run(debug=True)