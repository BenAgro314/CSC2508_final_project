#%%

import os
import sys
sys.path.append("/home/bagro/CSC2508_final_project/BLIP")

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from video_retrieval_models.blip_model import BlipBM25Model

#%% load blip model

model = BlipBM25Model(device=device)


#%% build index and query
video_path = "/home/bagro/videos/"
model.build_index(video_path)
video_path, frame = model.retrieve("whale")
print(video_path, frame)