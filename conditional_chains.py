from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a customer support agent"),
        ("human","classify the following review as either positive or negative or escalate: {review}")
    ]
)

positive_review_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human", "generate a thank you for a customer for his positive feedback on our product. feedback : {feedback}")
    ]
)

negativereview_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human", "generate a response message for negative feedback from the customer for our product. feedback : {feedback}")
    ]
)

escalate_review_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human", "generate a message to escalate this feed to a human agent. feedback : {feedback}")
    ]
)


branch_runnable = RunnableBranch(
    (
        lambda x : "positive" in x,
        positive_review_template | llm | StrOutputParser()
    ),
    (
        lambda x : "negative" in x,
        negativereview_template | llm | StrOutputParser()
    ),
    escalate_review_template | llm | StrOutputParser()

)

ex_1 = "your product mouse is the worst mouse that i have used so far, so much overpriced and no features at all. full false advertisements. mouse is not fit for drag clicking. i need a refund for this shit soon."
ex_2 = "the best keyboard that i have ever used. the keys are perfect and the sound produced is like a melody playing in my ears"
ex_3 = "not bad and not a good one too."

classify_chain = prompt_template| llm | StrOutputParser()

#chain = classify_chain | branch_runnable
chain = classify_chain | branch_runnable

result = chain.invoke({"review":ex_3})
print(result)