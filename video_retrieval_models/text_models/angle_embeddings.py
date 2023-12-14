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

    def __init__(self, device, pool = "max"):
        self.corpus = None
        self.index = None
        self.index_to_doc = None
        self.embedder = None
        self.device = device
        self.name = "angle"
        self.pool = pool

        self.cosine_sim = torch.nn.CosineSimilarity(dim=2)

    def build_index(self, text_dir_path: str) -> None:
        self.index_to_doc = {}

        self.angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        self.angle.set_prompt(prompt=Prompts.C)

        # self.angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')
        # self.angle.set_prompt(prompt=Prompts.A)


        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        running_index = 0
        # all_embeddings = []
        # self.doc_to_embedding_index = {}

        all_embeddings = []

        for doc in self.corpus:

            print(f"Indexing doc: {doc.name}")
            # sentence_list = [{'text': str(c)} for c in doc.captions]


            sentence_list = [{'text': str(doc.captions[i-1]) + str(doc.captions[i]) + str(doc.captions[i + 1])} for i in range(1,len(doc.captions)-2)]

            # text = ""
            # for c in doc.captions:
            #    text += str(c)
            # sentence_list = [{"text": text}]
            doc_embeddings = self.angle.encode(sentence_list, to_numpy=False)

            all_embeddings.append(doc_embeddings)

            # self.doc_to_embedding_index[doc.name] = (running_index, running_index + doc_embeddings.shape[0])
            # running_index += doc_embeddings.shape[0]
            # all_embeddings.append(doc_embeddings)

        max_doc_embedding_len = max(d.shape[0] for d in all_embeddings)

        self.all_embeddings = torch.stack(
            [
                torch.nn.functional.pad(
                    d, (0, 0, 0, max_doc_embedding_len - d.shape[0])
                ) for d in all_embeddings
            ],
            dim=0
        )
        self.valid_mask = ~torch.all(self.all_embeddings == 0, dim=-1) # (num_docs, num_per_doc)

    def retrieve(self, query: str, topk: int = 1) -> List[str]:
        query_embedding = self.angle.encode({'text': query}, to_numpy=False)
        cosine_sims = self.cosine_sim(
            query_embedding.reshape(1, 1, -1),
            self.all_embeddings # (num_docs, num_per_doc, F)
        ) # (num_docs, num_per_doc)

        if self.pool == "max":
            doc_scores = cosine_sims.max(dim=1)[0] # (num_docs)
        elif self.pool == "mean":
            counts = self.valid_mask.int().sum(dim=1) # (num_docs, )
            doc_scores = cosine_sims.sum(dim=1) / counts

        topk_inds = torch.topk(doc_scores, k=topk, dim=0).indices
        return [self.corpus[k.item()].name for k in topk_inds]
