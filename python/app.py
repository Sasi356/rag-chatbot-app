# from flask import Flask
# from flask import Flask, jsonify, request
# # from dotenv import load_dotenv
# # from langchain_community.vectorstores import Chroma
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import PromptTemplate


# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.core.memory import ChatMemoryBuffer
# from langchain.memory import ConversationSummaryMemory

# # from huggingface_hub import notebook_login
# # notebook_login()

# # load_dotenv()


# import os
# # from flask_cors import CORS


# app = Flask(__name__)
# # CORS(app)

# os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
# DB_FAISS_PATH = "C://Users//Infinitylearn//Desktop//local-langchain//python//"
# DATA_PATH = "C://Users//Infinitylearn//Desktop//local-langchain//python//knowledge//"

# @app.route('/', methods=['POST',"GET"])
# def chat():
#     if request.method == 'POST':
#         question = request.get_json()['value']
#         print("query in python:", question)
#         # db = build_vector_db()
#         chat_engine = build_retriever_chain()
#         res = chat_engine.invoke(question)
#         print("response: ", res)
#         return jsonify({'result': res})
#     elif request.method == 'GET':
#             response_body = {"name": "Nagato","about" :"Hello! I'm a full stack developer that loves python and javascript"}
#     else:    
#         response_body = "nothing here"
#     return response_body

# # @app.route('/document', methods =['POST', 'GET'])
# # def build_vector_db():
# #     documents= SimpleDirectoryReader(DATA_PATH).load_data()
# #     db = Chroma.from_documents(documents, OllamaEmbeddings(model="mxbai-embed-large", model_kwargs={'device': 'cpu'}))
# #     service_context=ServiceContext.from_defaults(chunk_size=1024,llm=llm,embed_model=embed_model)
# #     memory = ChatMemoryBuffer.from_defaults(token_limit=15000)

# #     index=VectorStoreIndex.from_documents(documents,service_context=service_context)
# #     chat_engine = index.as_chat_engine(chat_mode="context", memory=memory, system_prompt= system_prompt)

# #     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
# #     #                                                chunk_overlap=50)
# #     # texts = text_splitter.split_documents(documents)
# #     # print(texts)

# #     # db = FAISS.from_documents(texts, OllamaEmbeddings(model="mxbai-embed-large", model_kwargs={'device': 'cpu'}))
# #     # db.save_local(DB_FAISS_PATH)
# #     print(db)
# #     return db

# #     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# #     # documents = text_splitter.split_documents(docs)
#     # print(documents)
#     # db=FAISS.from_documents(documents[:1],OllamaEmbeddings(model="mxbai-embed-large"))
#     # db.save_local(DB_FAISS_PATH)

# def build_retriever_chain():
#     # prompt = ChatPromptTemplate.from_template("""
#     #     Answer the following question based only on the provided context. 
#     #     Think step by step before providing a detailed answer. 
#     #     I will tip you $1000 if the user finds the answer helpful. 
#     #     <context>
#     #     {context}
#     #     </context>
#     #     Question: {input}""")

#     template = """<s>[INST] Given the context - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]"""
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])

#     data_path = "C://Users//Infinitylearn//Desktop//local-langchain//python//knowledge//hanuman_pdf.pdf"
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=30, length_function=len,)

#     documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)
#     embed_model = OllamaEmbeddings(model="mxbai-embed-large")

#     vectordb = Chroma.from_documents(documents, embedding=embed_model)
#     rag = RetrievalQA.from_chain_type(
#             llm=Ollama(model="llama2:7b-chat-q2_K"),
#             retriever=vectordb.as_retriever(),
#             memory=ConversationSummaryMemory(llm = Ollama(model="llama2:7b-chat-q2_K")),
#             chain_type_kwargs={"prompt": prompt, "verbose": True},
#         )
#     print(rag)
#     return rag
#     # system_prompt="""
#     # You are a Q&A assistant. Your goal is to answer questions as
#     # accurately as possible based on the instructions and context provided.
#     # """

#     # # embedding = OllamaEmbeddings(model="mxbai-embed-large")
#     # # db = FAISS.load_local(DB_FAISS_PATH, embedding)
#     # # doc = {"data": db.page_content}
#     # print("db loaded")
#     # documents= SimpleDirectoryReader(DATA_PATH).load_data()
#     # # db = Chroma.from_documents(documents, OllamaEmbeddings(model="mxbai-embed-large", model_kwargs={'device': 'cpu'}))
#     # llm = Ollama(model="llama2:7b-chat-q2_K")
#     # embed_model = OllamaEmbeddings(model="mxbai-embed-large", model_kwargs={'device': 'cpu'})
#     # service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model=embed_model)
#     # memory = ChatMemoryBuffer.from_defaults(token_limit=15000)

#     # index=VectorStoreIndex.from_documents(documents,service_context=service_context)
#     # chat_engine = index.as_chat_engine(chat_mode="context", memory=memory, system_prompt= system_prompt)
#     # return chat_engine
#     # document_chain=create_stuff_documents_chain(llm,prompt)
#     # print("document chain loaded")

#     # retriever=db.as_retriever()
#     # retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     # print("retrieval chain loaded")
#     # return retrieval_chain

# #     output_parser=StrOutputParser()
# #     chain=llm|output_parser

# #     # llm_huggingface=HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0.,"max_length":128})
# #     output=chain.invoke(ques)
# #     print(output)
# #     return output

# if __name__ == '__main__':
#     app.run(debug=True)
