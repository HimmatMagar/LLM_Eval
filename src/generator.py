import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from retriever import build_retriever
from langchain_core.prompts import ChatPromptTemplate

retriever = build_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""Answer the question using only the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question: {question}
    Answer:"""

    return llm.invoke(prompt).content

ans = ask("What is the difference between an asset and a liability?")
print(ans)

# if __name__ == "__main__":
#     while True:
#         q = input("\nAsk a question (or 'exit'): ")
#         if q.lower() == "exit":
#             break
#         print("\n" + ask(q))