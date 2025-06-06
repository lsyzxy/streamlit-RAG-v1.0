from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma  # 导入Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dashscope import TextEmbedding
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

class TongyiEmbeddings(Embeddings):
    def __init__(self, api_key):
        # 直接使用传入的 api_key，而不是从环境变量获取
        self.api_key = api_key

    def embed_documents(self, texts):
        # 在调用 TextEmbedding.call 时传入 api_key
        return [TextEmbedding.call(input=text, model="text-embedding-v1", api_key=self.api_key).output["embeddings"][0]["embedding"] for text in texts]

    def embed_query(self, text):
        # 在调用 TextEmbedding.call 时传入 api_key
        return TextEmbedding.call(input=text, model="text-embedding-v1", api_key=self.api_key).output["embeddings"][0]["embedding"]

def qa_agent(qwen_api_key, dashscope_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=qwen_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    # 实例化 TongyiEmbeddings 时传入 dashscope_api_key
    embeddings_model = TongyiEmbeddings(dashscope_api_key)
    # 使用Chroma代替FAISS
    db = Chroma.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    chat_history = memory.load_memory_variables({})["chat_history"]
    response = qa.invoke({"chat_history": chat_history, "question": question})
    return response