import os

from types.document import Document

class Corpus:

    def __init__(self, docs_path: str = None):
        self.documents = []
        if docs_path is not None:
            doc_names = os.listdir(docs_path)
            for n in doc_names:
                doc = Document.read(os.path.join(docs_path, n))
                self.documents.append(doc)

    def add_document(self, doc: Document):
        self.documents.append(doc)

    def tokenize_documents(self) -> list[str]:
        """Tokenize documents"""
        return [d.tokenize() for d in self.documents]

    def __getitem__(self, id) -> Document:
        return self.documents[id]

    def __len__(self):
        return len(self.documents)

    def __str__(self):
        return f"Documents: {[d.name for d in self.documents]}"