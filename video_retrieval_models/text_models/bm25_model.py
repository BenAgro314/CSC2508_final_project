import numpy as np
from primitives.corpus import Corpus
from primitives.document import Document
from rank_bm25 import BM25Okapi
from video_retrieval_models.common import TextRetrievalProtocol 

class BM25Model(TextRetrievalProtocol):

    def __init__(self):
        self.corpus = None
        self.corpus_bm25 = None
        pass

    def build_index(self, text_dir_path: str) -> None:
        self.corpus = Corpus(text_dir_path)
        self.corpus_bm25 = BM25Okapi(self.corpus.tokenize_documents())

    def retrieve(self, query: str, topk: int = 1) -> list[tuple[Document, int]]:
        assert topk = 1, "BM25 does not support topk yet"
        assert self.corpus_bm25 is not None, "You have not called self.build_index() yet!"
        assert self.corpus is not None, "You have not called self.build_index() yet!"

        tokenized_query = query.split(" ")
        doc_scores = self.corpus_bm25.get_scores(tokenized_query)

        selected_doc = self.corpus[np.argmax(doc_scores)]
        doc_bm25 = BM25Okapi(selected_doc.tokenize_captions())
        caption_scores = doc_bm25.get_scores(tokenized_query)

        return [(selected_doc, np.argmax(caption_scores))]

