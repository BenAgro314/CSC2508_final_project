from primitives.corpus import Corpus
import json
from typing import List
import os
from pathlib import Path
from primitives.document import Document
import torch
from ordered_set import OrderedSet
from angle_emb import AnglE, Prompts

# TODO: document level search (heirarchical)
class AngleEmbeddings:

    def __init__(self, device):
        self.corpus = None
        self.index = None
        self.index_to_doc = None
        self.embedder = None
        self.device = device
        self.name = "angle"

        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)

    def build_index(self, text_dir_path: str) -> None:
        self.index_to_doc = {}

        self.angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        self.angle.set_prompt(prompt=Prompts.C)

        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        running_index = 0
        all_embeddings = []
        self.doc_to_embedding_index = {}
        for doc in self.corpus:

            print(f"Indexing doc: {doc.name}")
            sentence_list = [{'text': str(c)} for c in doc.captions]
            doc_embeddings = self.angle.encode(sentence_list, to_numpy=False)
            self.doc_to_embedding_index[doc.name] = (running_index, running_index + doc_embeddings.shape[0])
            running_index += doc_embeddings.shape[0]
            all_embeddings.append(doc_embeddings)

        self.all_embeddings = torch.concat(all_embeddings, dim=0)

    def retrieve(self, query: str, topk: int = 1) -> List[str]:
        query_embedding = self.angle.encode({'text': query}, to_numpy=False)

        cosine_sims = self.cosine_sim(
            query_embedding,
            self.all_embeddings
        )

        topk_inds = torch.topk(cosine_sims, k=topk, dim=0).indices


        return [self.index_to_doc[k.item()].name for k in topk_inds]

    def retrieve_unique(self, query: str, topk: int = 10) -> List:
        query_embedding = self.angle.encode({'text': query}, to_numpy=False)

        # best_docs = set()
        best_docs = OrderedSet([])
        search_topk = 200

        while len(best_docs) < topk:
            print(search_topk)
            cosine_sims = self.cosine_sim(
                query_embedding,
                self.all_embeddings
            )

            indices = torch.topk(cosine_sims, k=search_topk, dim=0).indices

            for ind in indices:
                for doc_name in self.doc_to_embedding_index:
                    if ind >= self.doc_to_embedding_index[doc_name][0] and ind < self.doc_to_embedding_index[doc_name][1]:
                        best_docs.add(doc_name)
                    
            search_topk *= 2

        out = list(best_docs[:topk])
        assert len(out) == topk
        return out

