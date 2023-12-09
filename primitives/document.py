import json
import os
from typing import List

from primitives.caption import Caption

class Document:
    def __init__(self, name: str, video_path: str, fps: int, frames: List[int] = None, captions: List[Caption] = None):
        self.name = name
        self.video_path = video_path
        self.fps = fps
        assert (frames is None) == (captions is None)
        if frames is not None:
            assert len(frames) == len(captions)
            self.frames = frames
            self.captions = captions
        else:
            self.frames = []
            self.captions = []

    def add_caption(self, frame: int, caption: Caption):
        assert isinstance(caption, Caption)
        assert len(self.frames) == len(self.captions)
        self.frames.append(frame)
        self.captions.append(caption)

    @classmethod
    def read(cls, path: str) -> 'Document':
        with open(path, "r") as f:
            data = json.load(f)
            return Document(
                name=data["name"],
                video_path=data["video_path"],
                fps=data["fps"],
                frames=data["frames"],
                captions=[Caption(s) for s in data["captions"]]
            )

    def save(self, path: str):
        with open(os.path.join(path, self.name + ".json"), "w") as f:
            data = {
                "name": self.name,
                "video_path": self.video_path,
                "fps": self.fps,
                "frames": self.frames,
                "captions": [str(s) for s in self.captions],
            }
            json.dump(data, f, indent=4)

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        st = ""
        i = 0
        for frame, caption in zip(self.frames, self.captions):
            time_str = f"{float(frame / self.fps):3.3f}"

            st += time_str
            st += " " * (20 - len(time_str))
            st += str(caption)
            i += 1
            if i != len(self):
                st += "\n"
        return st

    def tokenize(self):
        ret = []
        for cap in self.captions:
            ret += cap.tokenize()
        return ret


    def tokenize_captions(self) -> List[str]:
        return [c.tokenize() for c in self.captions]

    def __getitem__(self, id):
        return self.captions[id], self.frames[id]