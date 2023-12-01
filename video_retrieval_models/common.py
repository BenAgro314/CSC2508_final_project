from typing import Protocol

from primitives.document import Document

class VideoRetrievalProtocol(Protocol):

    def build_index(self, video_dir_path: str) -> None:
        ...

    def retrieve(self, query: str) -> tuple[str, int]:
        ...

class AudioToTextProtocol(Protocol):

    def build_index(self, audio_dir_path: str) -> None:
        ...

class VideoToTextProtocol(Protocol):

    def build_index(self, video_dir_path: str) -> None:
        ...

    @property
    def documents_path(self) -> str:
        pass

class TextRetrievalProtocol(Protocol):

    def build_index(self, text_dir_path: str) -> None:
        ...

    def retrieve(self, query: str, topk: int = 1) -> list[tuple[Document, int]]:
        ...


class BaseVideoRetrievalModel(VideoRetrievalProtocol):

    def __init__(self, video_to_text_model: VideoToTextProtocol, text_retrieval_model: TextRetrievalProtocol):
        self.video_to_text_model = video_to_text_model
        self.text_retrieval_model = text_retrieval_model

    def build_index(self, video_dir_path):
        self.video_to_text_model.build_index(video_dir_path)
        self.text_retrieval_model.build_index(self.video_to_text_model.documents_path)

    def retrieve(self, query: str, topk: int = 1) -> list[tuple[Document, int]]:
        return self.text_retrieval_model.retrieve(query, topk=topk)