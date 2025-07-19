from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import AIMessage,HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_path = os.path.join(current_dir, "chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = Chroma(persist_directory=persistent_path, embedding_function=embeddings)

query = "who came to pick up harry to hogwards for first time?"

retriver = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":3, "score_threshold":0.5}
)

relevent_docs = retriver.invoke(query)

for i, doc in enumerate(relevent_docs, 1):
    print(f"document {i} content: \n{doc.page_content}\n")