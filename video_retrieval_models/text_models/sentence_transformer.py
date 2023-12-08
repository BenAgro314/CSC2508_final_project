from primitives.corpus import Corpus
import json
import os
from pathlib import Path
from primitives.document import Document
from video_retrieval_models.common import TextRetrievalProtocol 
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
from ordered_set import OrderedSet

# TODO: document level search (heirarchical)
class SentenceTransformerModel(TextRetrievalProtocol):

    def __init__(self, device, model_name="all-MiniLM-L6-v2"):
        self.corpus = None
        self.index = None
        self.doc_to_embedding_index = None
        # self.embeddings = {}
        self.embedder = None
        self.model_name = model_name
        self.device = device
        # self.embedder = #SentenceTransformer(model_name).to(device)

    def load_checkpoint(self, faiss_path: str, json_path: str):
        self.index = faiss.read_index(faiss_path)
        with open(json_path, "r") as f:
            data = json.load(f)
            self.doc_to_embedding_index = data["doc_to_embedding_index"]
            self.model_name = data["model_name"]

    def save_checkpoint(self, faiss_path: str, json_path: str):
        faiss.write_index(self.index, faiss_path)
        with open(json_path, "w") as f:
            data = {
                "doc_to_embedding_index": self.doc_to_embedding_index,
                "model_name": self.model_name,
            }
            json.dump(data, f, indent=4)

    def build_index(self, text_dir_path: str) -> None:
        faiss_path = Path(text_dir_path).parent / "sentence_transformer.faiss"
        json_path = Path(text_dir_path).parent / "sentence_transformer.json"

        faiss_exists = os.path.exists(faiss_path)
        json_exists = os.path.exists(json_path)

        assert faiss_exists == json_exists
        # if the index exists, load it in
        if json_exists and faiss_exists:
            self.load_checkpoint(str(faiss_path), str(json_path))
        else:
            self.doc_to_embedding_index = {}

        self.embedder = SentenceTransformer(self.model_name).to(self.device)

        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        running_index = 0
        # all_embeddings = []
        for doc in self.corpus:
            if doc.name in self.doc_to_embedding_index:
                continue
            print(f"Indexing doc: {doc.name}")
            sentence_list = [str(c) for c in doc.captions]
            doc_embeddings = self.embedder.encode(sentence_list, convert_to_tensor=True)
            self.doc_to_embedding_index[doc.name] = (running_index, running_index + doc_embeddings.shape[0])
            running_index += doc_embeddings.shape[0]
            # all_embeddings.append(doc_embeddings)

            if self.index is None:
                self.index = faiss.IndexFlatIP(doc_embeddings.shape[-1])
            xb = doc_embeddings.cpu().numpy()
            faiss.normalize_L2(xb)
            self.index.add(xb)

        self.save_checkpoint(str(faiss_path), str(json_path))
        assert self.index is not None


    def retrieve(self, query: str, topk: int = 1) -> list[tuple[Document, int]]:
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        xq = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(xq)
        _, indices = self.index.search(xq, topk)
        indices = indices[0] # we only do one query at a time (currently)

        similarities = []

        for doc_name in self.doc_to_embedding_index:
            for ind in indices:
                if ind >= self.doc_to_embedding_index[doc_name][0] and ind < self.doc_to_embedding_index[doc_name][1]:
                    similarities += [(self.corpus.name_to_doc[doc_name], ind - self.doc_to_embedding_index[doc_name][0])]
                    
        return similarities

    def retrieve_unique(self, query: str, topk: int = 10) -> list:
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        xq = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(xq)

        # best_docs = set()
        best_docs = OrderedSet([])
        search_topk = int(2 * topk)

        while len(best_docs) < topk:
            _, indices = self.index.search(xq, search_topk)
            indices = indices[0] # we only do one query at a time (currently)

            for ind in indices:
                for doc_name in self.doc_to_embedding_index:
                    if ind >= self.doc_to_embedding_index[doc_name][0] and ind < self.doc_to_embedding_index[doc_name][1]:
                        best_docs.add(doc_name)
                    
            search_topk += int(2 * topk)

        out = list(best_docs[:topk])
        assert len(out) == topk
        return out

if __name__ == "__main__":
    m = SentenceTransformerModel("cuda")
    m.build_index("/home/bagro/llava_documents/")
    print(m.retrieve("tiger", topk=3))