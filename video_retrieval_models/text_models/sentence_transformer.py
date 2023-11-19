from primitives.corpus import Corpus
from primitives.document import Document
from video_retrieval_models.common import TextRetrievalProtocol 
from sentence_transformers import SentenceTransformer, util
import torch

# TODO: caching and vector DB
# TODO: document level search (heirarchical)
class SentenceTransformerModel(TextRetrievalProtocol):

    def __init__(self, device, model_name="all-MiniLM-L6-v2"):
        self.corpus = None
        self.embeddings = {}
        self.embedder = SentenceTransformer(model_name).to(device)


    def build_index(self, text_dir_path: str) -> None:
        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        max_seq_length = self.embedder.max_seq_length
        for doc in self.corpus:
            sentence_list = [str(c) for c in doc.captions]
            doc_embeddings = self.embedder.encode(sentence_list, convert_to_tensor=True) #, max_length=max_seq_length)
            self.embeddings[doc] = doc_embeddings

    def retrieve(self, query: str, topk: int = 1) -> list[tuple[Document, int]]:
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        similarities = []

        for doc in self.embeddings:
            cos_scores = util.cos_sim(query_embedding, self.embeddings[doc])[0]
            # top_results = torch.topk(cos_scores, k=1)
            # doc_similarities.append(torch.max(cos_scores))
            similarities += [(cos_scores[i].item(), doc, i) for i in range(len(doc))]

        similarities = sorted(similarities, key = lambda x: x[0], reverse=True)
        similarities[:topk]

        return [(t[1], t[2]) for t in similarities[:topk]]


if __name__ == "__main__":
    m = SentenceTransformerModel("cuda")
    m.build_index("/home/bagro/llava_documents/")
    print(m.retrieve("tiger", topk=3))