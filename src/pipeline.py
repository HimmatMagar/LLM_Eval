from generator import ask
from retriever import build_retriever

class Pipeline:
    def __init__(self):
        self.retriever = build_retriever()

    def invoke(self, query):
        docs = self.retriever.invoke(query)

        context = [docs.page_content for docs in docs]

        ans = ask(question=query, context=context)

        return {
            'query': query,
            'context': context,
            'answer': ans
        }


if __name__ == "__main__":
    rag = Pipeline()
    result = rag.invoke(query="What is the difference between an asset and a liability?")
    print("Query", result['query'])
    print("Answer", result['answer'])
    print("Context chunks")
    for i, chunk in enumerate(result['context']):
        print(f". [{i}] {chunk[:120]} ")