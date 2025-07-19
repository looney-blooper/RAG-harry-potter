from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"hp1.txt")
presistent_dir = os.path.join(current_dir,"chroma_db")

if not os.path.exists(presistent_dir):
    if not os.path.exists(file_path):
        raise FileNotFoundError("no dataset is found")
    
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("no. of chucks : "+str(len(docs)))

    embedder = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    db = Chroma.from_documents(docs, embedder, persist_directory=presistent_dir)

else:
    print("vector db already present!.")