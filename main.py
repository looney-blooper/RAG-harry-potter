from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

chat_history=[]
system_message = SystemMessage(content="You are a helpful AI assistant")
chat_history.append(system_message)

while True:
    query = input("You : ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result))
    print("AI :"+result)

print("the AI is offline now.")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print(embeddings.embed_query("hello, world!"))

    