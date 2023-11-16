#%%

import os
import sys
from video_retrieval_models.text_models.bm25_model import BM25Model
from video_retrieval_models.common import BaseVideoRetrievalModel

from video_retrieval_models.video_models.llava_model import LlavaModel
sys.path.append("/home/bagro/CSC2508_final_project/BLIP")

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from video_retrieval_models.blip_bm25_model import BlipBM25Model

#%% load blip model

# model = BlipBM25Model(device=device)
model = BaseVideoRetrievalModel(
    LlavaModel(device),
    BM25Model()
)


#%% build index and query
video_path = "/home/bagro/videos/"
model.build_index(video_path)
video_path, frame = model.retrieve("whale")
print(video_path, frame)

#%%
from IPython.display import HTML
from base64 import b64encode
import cv2

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
mp4 = open(video_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
start_time = frame / fps
end_time = start_time + 5
HTML(f"""
<video width=400 controls autoplay>
      <source src="{data_url}#t={start_time},{end_time}" type="video/mp4">
</video>
""")