from primitives.corpus import Corpus
import json
import time
from typing import List
import numpy as np
import os
from pathlib import Path
from primitives.document import Document
import torch
from ordered_set import OrderedSet
from angle_emb import AnglE, Prompts
from llama_cpp import Llama

# TODO: document level search (heirarchical)
class DocLevelAngleEmbeddingsReRank:

    def __init__(self, device, model_name=None, pool="mean"):
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
        self.llm = Llama(model_path="/home/ubuntu/csc2508/models/llama-2/llama-2-13b.Q5_K_M.gguf", n_gpu_layers=-1, logits_all=True, n_ctx=2048) # , n_ctx=0)

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

        # True token: 
        self.true_token = self.llm.tokenize(bytes("True", "utf-8"))[1]
        # False token
        self.false_token = self.llm.tokenize(bytes("False", "utf-8"))[1]

    def retrieve(self, query: str, topk: int = 1) -> List[str]:
        query_embedding = self.angle.encode({'text': query}, to_numpy=False)
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1).reshape(1, -1)

        cosine_sims = self.cosine_sim(
            query_embedding,
            self.all_embeddings
        )

        topk_inds = torch.topk(cosine_sims, k=topk, dim=0).indices

        # re-ranking
        topk_docs = [self.index_to_doc[k.item()] for k in topk_inds]
        doc_scores = []
        print(f"Before: {[d.name for d in topk_docs]}")
        for doc in topk_docs:
            t = time.time()
            self.llm.reset()
            preamble = "The following is a transcript of the images in a video:\n"
            preamble_tokens = self.llm.tokenize(bytes(preamble, "utf-8"))
            video_text = ""
            for c in doc.captions:
                video_text += str(c)
                video_text += "\n"
            video_text_tokens = self.llm.tokenize(bytes(video_text, "utf-8"))[1:]
            question_text = f"Q: True of False, this is a good description of the video: {query}. A: True"
            question_text_tokens = self.llm.tokenize(bytes(question_text, "utf-8"))[1:]

            if len(preamble_tokens) + len(video_text_tokens) + len(question_text_tokens) > self.llm.n_ctx():
                video_text_tokens = video_text_tokens[:self.llm.n_ctx() - (len(preamble_tokens) + len(question_text_tokens))]
            tokens = preamble_tokens + video_text_tokens + question_text_tokens
            assert len(tokens) <= self.llm.n_ctx()
            self.llm.eval(tokens)
            doc_score = self.llm.scores[len(tokens)-2, self.true_token]
            doc_scores.append(-doc_score) # because we want the highest scores first in the sort
            print(f"Time to re-rank doc: {time.time() - t}")
        re_ordering = np.argsort(doc_scores)
        res = [topk_docs[i].name for i in re_ordering]
        print(f"After: {res}")
        return res

