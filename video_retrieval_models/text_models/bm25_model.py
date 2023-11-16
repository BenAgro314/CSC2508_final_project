import numpy as np
from primitives.corpus import Corpus
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

    def retrieve(self, query: str) -> tuple[str, int]:
        assert self.corpus_bm25 is not None, "You have not called self.build_index() yet!"
        assert self.corpus is not None, "You have not called self.build_index() yet!"

        tokenized_query = query.split(" ")
        doc_scores = self.corpus_bm25.get_scores(tokenized_query)

        selected_doc = self.corpus[np.argmax(doc_scores)]
        doc_bm25 = BM25Okapi(selected_doc.tokenize_captions())
        caption_scores = doc_bm25.get_scores(tokenized_query)
        selected_caption = selected_doc[np.argmax(caption_scores)]

        return selected_doc.video_path, selected_caption[1]

