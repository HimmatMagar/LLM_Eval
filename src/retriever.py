from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_docs():
    documents = []

    for path in Path("../data/").rglob('*'):
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
    pass

docs = load_docs()
print(docs[20])