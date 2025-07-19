from langchain.schema.runnable import RunnableSequence, RunnableLambda
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        ("system","you are a movie critic"),
        ("human","explain the plot of {movie} in 1 line"),
    ]
)

format_prompt_1 = RunnableLambda(lambda x : prompt_template_1.format_prompt(**x))
invoke_model = RunnableLambda(lambda x : llm.invoke(x.to_messages()))
get_plot = RunnableLambda(lambda x : x)


prompt_template_2 = ChatPromptTemplate.from_messages(
    [
        ("system","you are a science enthusiast"),
        ("human","rate the movie plot regards to the science taste: {plot} , out of 5 stars")
    ]
)

format_prompt_2 = RunnableLambda(lambda x : prompt_template_2.format_prompt(**x))

prepare_format_2 = RunnableLambda(lambda x : {"plot":x})

chain = prompt_template_1 | llm | StrOutputParser() | prepare_format_2 | prompt_template_2 | llm | StrOutputParser()

response = chain.invoke({"movie":"ritual"})
print(response)