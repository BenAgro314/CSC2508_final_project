from typing import List
import os
import numpy as np
import torch

from video_retrieval_models.video_models.generate_llava_embeddings import get_llava_text_embedding, make_model


class LlavaEmbeddingsRetrieval:

    def __init__(self):
        self.clip_ctx, self.llava = make_model()
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.index_to_doc = {}
        self.corpus = set()

    def build_index(self, embedding_dir_path: str) -> None:
        self.index_to_doc = {}
        all_embeddings = []
        existing_embedding_names = os.listdir(embedding_dir_path)
        for ind, embedding_file in enumerate(existing_embedding_names):
            embedding_path = os.path.join(embedding_dir_path, embedding_file)
            embeddings_dict = np.load(embedding_path)

            embeddings_list = []

            for frame_number in embeddings_dict.keys():
                frame_embedding = torch.tensor(embeddings_dict[frame_number]).to("cuda")
                embeddings_list.append(frame_embedding)

            video_embedding = torch.stack(embeddings_list, dim=0).mean(dim=0) # mean pooling
            all_embeddings.append(video_embedding)
            self.index_to_doc[ind] = os.path.splitext(embedding_file)[0]
            self.corpus.add(self.index_to_doc[ind])

        self.all_embeddings = torch.stack(all_embeddings, dim=0)

    def retrieve(self, query: str, topk: int = 1) -> List[str]:
        self.llava.reset()
        query_embedding = get_llava_text_embedding(self.llava, query)[None]
        cosine_sims = self.cosine_sim(
            query_embedding,
            self.all_embeddings
        )
        topk_inds = torch.topk(cosine_sims, k=topk, dim=0).indices
        return [self.index_to_doc[k.item()] for k in topk_inds]

