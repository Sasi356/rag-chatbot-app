from flask import Flask
from flask import Flask, jsonify, request
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts.prompts import SimpleInputPrompt
from io import BytesIO
from transformers import BitsAndBytesConfig
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from llama_index.llms.huggingface import HuggingFaceLLM

from pypdf import PdfReader, PdfWriter

import os
# from flask_cors import CORS


app = Flask(__name__)
# CORS(app)

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
#uploaded file is stored here
DATA_PATH = "C://Users//Infinitylearn//Desktop//local-langchain//python//knowledge//"
system_prompt=""" 
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
    """
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files["file_from_react"]
        reader = PdfReader(file)
        page = reader.pages[0]
        print('-----------------------------')
        print('received a file')
        print('-----------------------------')
        print(page.extract_text())
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(DATA_PATH + "temp.pdf")
        return "File uploaded succesfully"
    except:
        return "Failed to upload"


@app.route('/chat', methods=['POST',"GET"])
def chat():
    if request.method == 'POST':
        # #testcode
        # return "nothing special"
        question = request.get_json()['value']
        service_context = configure_models()
        chat_engine = build_chat_engine(service_context)
        res = chat_engine.chat(question)
        print("response: ", res)
        return str(res)
    elif request.method == 'GET':
            response_body = {"name": "Sasi"}
    else:    
        response_body = "nothing here"
    return response_body

def configure_models():

    query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
    
    # quantization from transformers not working on cpu
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    llm = Ollama(model="llama2:7b-chat-q2_K")
    embed_model = OllamaEmbeddings(model="all-minilm")

    service_context=ServiceContext.from_defaults(
        chunk_size=512,
        llm=llm,
        embed_model=embed_model
    )

    return service_context

def build_chat_engine(service_context):

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    documents= SimpleDirectoryReader(DATA_PATH).load_data()
    index=VectorStoreIndex.from_documents(documents, service_context=service_context)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt= system_prompt
    )
    return chat_engine

if __name__ == '__main__':
    app.run(debug=True)
