from primitives.corpus import Corpus
import json
import os
from pathlib import Path
from primitives.document import Document
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import faiss
from ordered_set import OrderedSet

# TODO: document level search (heirarchical)
class DocLevelSentenceTransformerModel:

    def __init__(self, device, model_name="all-MiniLM-L6-v2", pool = "mean", use_cosine: bool = True, use_l2: bool = False):
        self.corpus = None
        self.index = None
        self.index_to_doc = None
        # self.embeddings = {}
        self.embedder = None
        self.model_name = model_name
        self.device = device
        self.pool = pool
        # self.embedder = #SentenceTransformer(model_name).to(device)
        self.name = "doc_level_sentence_transformer"
        self.use_cosine = use_cosine
        self.use_l2 = use_l2

        if self.use_l2:
            assert not self.use_cosine

    def load_checkpoint(self, faiss_path: str, json_path: str):
        self.index = faiss.read_index(faiss_path)
        with open(json_path, "r") as f:
            data = json.load(f)
            self.index_to_doc = data["index_to_doc"]
            self.model_name = data["model_name"]

    def save_checkpoint(self, faiss_path: str, json_path: str):
        faiss.write_index(self.index, faiss_path)
        with open(json_path, "w") as f:
            data = {
                "index_to_doc": self.index_to_doc,
                "model_name": self.model_name,
            }
            json.dump(data, f, indent=4)

    def build_index(self, text_dir_path: str) -> None:
        name = self.name + "_" + self.model_name + "_" + self.pool + "_" + str(self.use_cosine) + "_" + str(self.use_l2)

        faiss_path = Path(text_dir_path).parent / f"{name}.faiss"
        json_path = Path(text_dir_path).parent / f"{name}.json"

        faiss_exists = os.path.exists(faiss_path)
        json_exists = os.path.exists(json_path)

        assert faiss_exists == json_exists
        # if the index exists, load it in
        if json_exists and faiss_exists:
            self.load_checkpoint(str(faiss_path), str(json_path))
        else:
            self.index_to_doc = {}

        self.embedder = SentenceTransformer(self.model_name).to(self.device)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        running_index = max(self.index_to_doc.keys()) if len(self.index_to_doc) > 0 else 0
        # all_embeddings = []
        for doc in self.corpus:
            if doc.name in set(self.index_to_doc.values()):
                continue
            print(f"Indexing doc: {doc.name}")
            sentence_list = [str(c) for c in doc.captions]
            doc_embeddings = self.embedder.encode(sentence_list, convert_to_tensor=True)
            if self.pool == "mean":
                agg_doc_embedding = doc_embeddings.mean(dim=0).reshape(1, -1)
            elif self.pool == "max":
                agg_doc_embedding = doc_embeddings.max(dim=0)[0].reshape(1, -1)
            self.index_to_doc[str(running_index)] = doc.name
            running_index += 1
            if self.index is None:
                if self.use_l2:
                    self.index = faiss.IndexFlatL2(agg_doc_embedding.shape[-1])
                else:
                    self.index = faiss.IndexFlatIP(agg_doc_embedding.shape[-1])
            xb = agg_doc_embedding.cpu().numpy()
            if self.use_cosine:
                faiss.normalize_L2(xb)
            self.index.add(xb)

        self.save_checkpoint(str(faiss_path), str(json_path))
        assert self.index is not None

    def retrieve(self, query: str, topk: int = 1):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        xq = query_embedding.cpu().numpy().reshape(1, -1)
        if self.use_cosine:
            faiss.normalize_L2(xq)
        _, indices = self.index.search(xq, topk)
        indices = indices[0] # we only do one query at a time (currently)

        docs = []

        for ind in indices:
            docs.append(self.index_to_doc[str(ind)])

        return docs

        # # re-ranking
        # to_sort = []
        # for ind in indices:
        #     doc_name = self.index_to_doc[str(ind)]
        #     doc = self.corpus.name_to_doc[doc_name]

        #     scores = self.cross_encoder.predict(
        #         [[str(c), query] for c in doc.captions],
        #         convert_to_tensor=True,
        #     )
        #     max_score = scores.max()
        #     to_sort.append((max_score, doc.name))
        #     #docs.append(self.index_to_doc[str(ind)])

        # res = sorted(to_sort, key=lambda x: -x[0])

        # return [t[1] for t in res]


if __name__ == "__main__":
    m = DocLevelSentenceTransformerModel("cuda")
    m.build_index("/home/bagro/llava_documents/")
    print(m.retrieve("tiger", topk=3))