from typing import Protocol


class VideoRetrievalModel(Protocol):

    def initialize(self, video_dir_path: str) -> None:
        ...

    def retrieve(self, query: str) -> tuple[str, int]:
        ...