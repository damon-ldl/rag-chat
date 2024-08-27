from langchain_community.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever 
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from configs.model_config import EMBEDDING_MODEL
 
import os
 
# 连接本地部署的OpenAI服务
model = ChatOpenAI(
    streaming=True,
    verbose=True,
    callbacks=[],
    openai_api_key="your key",
    openai_api_base="https://api.moonshot.cn/v1",
    model_name="moonshot-v1-8k",
    temperature=0
)
 
# 加载Documents
base_dir = './files' # 文档的存放目录
documents = []
for file in os.listdir(base_dir): 
    # 完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path,encoding="utf-8")
        documents.extend(loader.load())
 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10) # 文档分割器
chunked_documents = text_splitter.split_documents(documents)
 
# 创建 embeedings sentence-transformers提供向量表示
embeddings = HuggingFaceEmbeddings(model_name="D:\\project\\LangChain-ChatGLM-Webui\\model\\text2vec-base-chinese")
vectorstore = FAISS.from_documents(chunked_documents, embeddings)

 

 
# 构建一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=model)
print(retriever_from_llm)
 
# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(model, retriever=retriever_from_llm)
result = qa_chain("你是谁")
 
print(result)
