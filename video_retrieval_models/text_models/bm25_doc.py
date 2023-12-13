import numpy as np
from primitives.corpus import Corpus
from primitives.document import Document
from rank_bm25 import BM25Okapi
import torch

class BM25ModelDocRetrieval:

    def __init__(self):
        self.corpus = None
        self.corpus_bm25 = None
        pass

    def build_index(self, text_dir_path: str) -> None:
        self.corpus = Corpus(text_dir_path)
        self.corpus_bm25 = BM25Okapi(self.corpus.tokenize_documents())

    def retrieve(self, query: str, topk: int = 1):
        assert self.corpus_bm25 is not None, "You have not called self.build_index() yet!"
        assert self.corpus is not None, "You have not called self.build_index() yet!"

        tokenized_query = query.split(" ")
        doc_scores = torch.tensor(self.corpus_bm25.get_scores(tokenized_query))

        topk_inds = torch.topk(doc_scores, k=topk, dim=0).indices

        return [self.corpus[k.item()].name for k in topk_inds]

