import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from models.blip import blip_decoder
from primitives.caption import Caption
from primitives.corpus import Corpus
from primitives.document import Document
from rank_bm25 import BM25Okapi
from utils import load_video_into_images

from video_retrieval_models.common import VideoRetrievalModel


class BlipBM25Model(VideoRetrievalModel):

    def __init__(self, device: torch.device):
        self.image_size = 384
        self.model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

        self.model = blip_decoder(pretrained=self.model_url, image_size=self.image_size, vit='base', med_config="BLIP/configs/med_config.json")
        self.model.eval()
        self.model = self.model.to(device)
        self.device=device
        self.process_fps = 2
        self.corpus = None

    def initialize(self, video_dir_path: str) -> None:
        video_path = video_dir_path
        doc_path = str(Path(video_dir_path).parent / "documents")

        videos = {Path(v).stem: v for v in os.listdir(video_path)}
        docs = {Path(d).stem: d for d in os.listdir(doc_path)}

        assert len(docs) <= len(videos), f"Documents {set(docs.keys()).difference(set(videos.keys()))} do not have corresponding videos"
        unprocessed_vids = set(videos.keys()).difference(set(docs.keys()))
        print(f"Unprocessed videos: {unprocessed_vids}")

        for vid_name in unprocessed_vids:
            vid_path = os.path.join(video_path, videos[vid_name])
            print(f"Processing {vid_name}")

            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            document = Document(name=vid_name, video_path=vid_path, fps=fps)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Video fps: {fps}, total frames: {total_frames}")
            subsample_rate = int(round(fps / self.process_fps)) # we want to read in videos at 2 fps

            with torch.no_grad():
                for frame, img, pil_image in tqdm.tqdm(load_video_into_images(vid_path, self.image_size, self.device, subsample_rate)):

                    caption = self.model.generate(img, sample=False, num_beams=3, max_length=20, min_length=5)
                    document.add_caption(frame=frame, caption=Caption(caption[0]))

            print(document)
            document.save(doc_path)

        self.corpus = Corpus(doc_path)

    def retrieve(self, query: str) -> tuple[str, int]:
        assert self.corpus is not None, "You have not called self.initialize() yet!"

        tokenized_query = query.split(" ")
        corpus_bm25 = BM25Okapi(self.corpus.tokenize_documents())
        doc_scores = corpus_bm25.get_scores(tokenized_query)

        selected_doc = self.corpus[np.argmax(doc_scores)]

        doc_bm25 = BM25Okapi(selected_doc.tokenize_captions())
        caption_scores = doc_bm25.get_scores(tokenized_query)
        selected_caption = selected_doc[np.argmax(caption_scores)]

        return selected_doc.video_path, selected_caption[1]
