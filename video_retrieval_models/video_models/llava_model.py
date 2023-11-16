
from video_retrieval_models.common import VideoToTextProtocol, TextRetrievalProtocol, BaseVideoRetrievalModel
import torch

class BlipModel(VideoToTextProtocol):

    def __init__(self, device: torch.device):
        pass

    def build_index(self, video_dir_path: str) -> None:
        pass

    @property
    def documents_path(self) -> str:
        return self.doc_path
