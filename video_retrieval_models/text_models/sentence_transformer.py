from primitives.corpus import Corpus
from primitives.document import Document
from video_retrieval_models.common import TextRetrievalProtocol 
from sentence_transformers import SentenceTransformer, util
import torch

# TODO: caching and vector DB
# TODO: document level search (heirarchical)
class SentenceTransformerModel(TextRetrievalProtocol):

    def __init__(self, device, model_name="sentence-t5-large"):
        self.corpus = None
        self.embeddings = {}
        self.embedder = SentenceTransformer(model_name).to(device)


    def build_index(self, text_dir_path: str) -> None:
        self.corpus = Corpus(text_dir_path)
        sentence_list = []
        for doc in self.corpus:
            sentence_list = [str(c) for c in doc.captions]
            for c in doc.captions:
                print(c, len(c))
            doc_embeddings = self.embedder.encode(sentence_list, convert_to_tensor=True)
            self.embeddings[doc] = doc_embeddings

    def retrieve(self, query: str) -> tuple[str, int]:
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        # find best matching sentence (super naive)
        doc_similarities = []
        docs = [doc for doc in self.embeddings]
        similarities = {}

        for doc in docs:
            cos_scores = util.cos_sim(query_embedding, self.embeddings[doc])[0]
            # top_results = torch.topk(cos_scores, k=1)
            doc_similarities.append(torch.max(cos_scores))
            similarities[doc] = cos_scores

        selected_doc = docs[torch.argmax(torch.stack(doc_similarities, dim=0)).item()]
        best_frame = torch.argmax(similarities[selected_doc]).item()
        selected_caption = selected_doc[best_frame]

        return selected_doc.video_path, selected_caption[1]


if __name__ == "__main__":
    m = SentenceTransformerModel("cuda")
    m.build_index("/home/bagro/documents/")
    print(m.retrieve("a"))