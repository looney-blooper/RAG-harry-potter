from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import HumanMessage ,AIMessage ,SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

def plot_movie(plot):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system","you are a movie critic"),
            ('human',"explain the plot of the movie {movie} in one line")
        ]
    )
    return prompt_template.format_prompt(movie=plot)

def teacher_analysis(plot):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system","you are a high school teacher"),
            ("human","analyse the movie {movie}"),
        ]
    )
    return prompt_template.format_prompt(movie=plot)

plot_movie_chain = RunnableLambda(lambda x : plot_movie(x)) | llm | StrOutputParser()
teacher_analysis_chain = RunnableLambda(lambda x : teacher_analysis(x)) | llm | StrOutputParser()

should_telecast_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a school principal"),
        ("human","a movie has a plot of {plot}\nand teachers has analysed it as {t_analysis}. can this movie played in the auditorium for hostel students on the movie day?")
    ]
)

recommend_movie_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a science and tech entusiast who loves war movies"),
        ("human","recommend only one movie to watch")
    ]
)

chain = (
    recommend_movie_template
    | llm
    | StrOutputParser()
    | RunnableParallel(branches = {"plot": plot_movie_chain, "t_ana":teacher_analysis_chain})
    | RunnableLambda( lambda x : should_telecast_template.format_prompt(plot=x["branches"]["plot"],t_analysis=x["branches"]["t_ana"]))
    | llm | StrOutputParser()
)

response = chain.invoke({})
print(response)