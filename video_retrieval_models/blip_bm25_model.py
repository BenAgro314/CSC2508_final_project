import torch
from video_retrieval_models.common import BaseVideoRetrievalModel
from video_retrieval_models.text_models.bm25_model import BM25Model
from video_retrieval_models.video_models.blip_model import BlipModel


class BlipBM25Model(BaseVideoRetrievalModel):

    def __init__(self, device: torch.device):
        video_model = BlipModel(device)
        text_model = BM25Model()
        super().__init__(video_model, text_model)