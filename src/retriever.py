import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
BASE = Path(__file__).resolve().parent.parent
DB_DIR = str(BASE / "chroma_store")
DATA_DIR = BASE / "data"

def load_docs():
    documents = []

    for path in DATA_DIR.rglob('*'):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file_path=str(path))
            docs = loader.load()
        else:
            continue

        for doc in docs:
            doc.metadata['source'] = str(path)
            doc.metadata['file_name'] = path.name

        documents.extend(docs)
    return documents


def load_store():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embedding)

    docs = load_docs()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
    ).split_documents(docs)

    # Then pass in the cleaned list
    return Chroma.from_documents(chunks, embedding, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("Who is the author of Rich Dad Poor Dad?")
    
    for r in results:
        print(f"[Page {r.metadata['page_label']}] {r.page_content[:150]}...\n")