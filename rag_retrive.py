from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_path = os.path.join(current_dir, "chroma_db")

llm = GoogleGenerativeAI(model="gemini-2.5-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = Chroma(persist_directory=persistent_path, embedding_function=embeddings)

user_inputs = input("Enter your query: ")

retriver = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":9, "score_threshold":0.4}
)

relevent_docs = retriver.invoke(user_inputs)

query = ("Here are some documents that might help to answer the question: " 
        +user_inputs
        +"\nrelevent documents :\n"
        +"\n\n".join([doc.page_content for doc in relevent_docs])
        +"provide a rough answer based on the given document only"
        )

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpfull assistant"),
        ("human", "{query}")
    ]
)

chain = prompt_template | llm | StrOutputParser()

response = chain.invoke({"query":query})
print(response)

"""for r in relevent_docs:
    print(r.page_content)
    print()"""