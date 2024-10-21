from Database import vectordb
from flask import Flask
from flask_cors import CORS
Application = Flask(__name__)
CORS(Application)


@Application.route("/")
def homepage():
    return "Home Route"


@Application.route("/Upload_files", methods=['POST'])
def upload_files():
    input_file_path = "Pdf_files"
    v_database = vectordb(input_file_path)
    v_database.persist()
    return v_database


@Application.route("/Enter_Query", methods=['POST'])
def enter_query():
    return "home"


if "__main__" == __name__:
    Application.run(debug=True)