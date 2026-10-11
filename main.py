from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Which attractions should I visit during my first day in Budapest?
# What public transport options are available in Budapest?
# What should tourists check before visiting an attraction?
# I enjoy walking and scenic views. What places would you recommend, and how should I plan my day?
# What is the population of Budapest in 2026?

MODEL_NAME = 'gpt-4o-mini'

load_dotenv()
api_key = os.getenv('API_KEY')

embedding_function = OpenAIEmbeddings(api_key=api_key, model='text-embedding-3-small')

vectorstore = Chroma(
    collection_name='budapest_attractions',
    embedding_function=embedding_function,
    persist_directory='./travel_db'
)

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)


message = """
Answer the question using only the provided context.
If the answer is not explicitly stated in the context,
say "I don't know."

Context:
{context}

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_messages([('human', message)])
llm = ChatOpenAI(model=MODEL_NAME, api_key=api_key)

rag_chain = ({'context': retriever, 'question': RunnablePassthrough()}) | prompt_template | llm

while True:
    user_question = input('YOU: ')

    question = "What should tourists check before visiting an attraction?"

    res = rag_chain.invoke(user_question)

    print("TRAVEL_BOT: ", res.content)

    print('<==================================>')

    results = retriever.invoke(user_question)

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Source:", doc.metadata.get("source"))


