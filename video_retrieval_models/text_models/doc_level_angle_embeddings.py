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
class DocLevelAngleEmbeddings:

    def __init__(self, device, pool="mean"):
        self.corpus = None
        self.index = None
        self.index_to_doc = None
        # self.embeddings = {}
        self.embedder = None
        self.device = device
        self.pool = pool
        self.name = "angle"

        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)

    def build_index(self, text_dir_path: str) -> None:
        name = self.name  + "_" + self.pool

        # faiss_path = Path(text_dir_path).parent / f"{name}.faiss"
        # json_path = Path(text_dir_path).parent / f"{name}.json"

        # faiss_exists = os.path.exists(faiss_path)
        # json_exists = os.path.exists(json_path)

        # assert faiss_exists == json_exists
        # # if the index exists, load it in
        # if json_exists and faiss_exists:
        #     self.load_checkpoint(str(faiss_path), str(json_path))
        # else:
        self.index_to_doc = {}

        # self.embedder = (self.model_name).to(self.device)
        self.angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        self.angle.set_prompt(prompt=Prompts.C)
        # self.angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')
        # self.angle.set_prompt(prompt=Prompts.A)

        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        running_index = 0
        all_embeddings = []
        for doc in self.corpus:

            print(f"Indexing doc: {doc.name}")
            sentence_list = [{'text': str(c)} for c in doc.captions]
            doc_embeddings = self.angle.encode(sentence_list, to_numpy=False)
            if self.pool == "mean":
                agg_doc_embedding = doc_embeddings.mean(dim=0).reshape(1, -1)
            elif self.pool == "max":
                agg_doc_embedding = doc_embeddings.max(dim=0)[0].reshape(1, -1)

            self.index_to_doc[running_index] = doc

            running_index += 1

            all_embeddings.append(
                torch.nn.functional.normalize(agg_doc_embedding, dim=-1)
            )

        self.all_embeddings = torch.concat(all_embeddings, dim=0)

    def retrieve(self, query: str, topk: int = 1) -> List[str]:
        query_embedding = self.angle.encode({'text': query}, to_numpy=False)
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1).reshape(1, -1)

        cosine_sims = self.cosine_sim(
            query_embedding,
            self.all_embeddings
        )

        topk_inds = torch.topk(cosine_sims, k=topk, dim=0).indices


        return [self.index_to_doc[k.item()].name for k in topk_inds]

