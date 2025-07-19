from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"books")
presistent_dir = os.path.join(current_dir,"chroma_db")

if not os.path.exists(presistent_dir):
    if not os.path.exists(file_path):
        raise FileNotFoundError("no dataset is found")
    
    books = [ doc for doc in os.listdir(file_path) if doc.endswith(".txt")]

    documents = []
    for book in books:
        book_path = os.path.join(file_path,book)
        loader = TextLoader(book_path)
        book_docs=loader.load()
        for doc in book_docs:
            doc.metadata = {"source":book}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("no. of chucks : "+str(len(docs)))

    embedder = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    db = Chroma.from_documents(docs, embedder, persist_directory=presistent_dir)

else:
    print("vector db already present!.")