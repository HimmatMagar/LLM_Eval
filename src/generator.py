import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from retriever import build_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = build_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            """
            Answer the question using only the context below.
            If the answer is not in the context, say "I don't know".

            Context: {context}
            """
        ),
        (
            "human",
            "Question: {question}"
        )
    ])

chain = prompt | llm | StrOutputParser()

def ask(question: str, context: list[str]):
    context = "\n\n".join(context)
    return chain.invoke({"question": question, "context": context})

ans = ask("What is the difference between an asset and a liability?")
print(ans)

# if __name__ == "__main__":
#     while True:
#         q = input("\nAsk a question (or 'exit'): ")
#         if q.lower() == "exit":
#             break
#         print("\n" + ask(q))