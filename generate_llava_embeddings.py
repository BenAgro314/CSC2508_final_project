from llama_cpp import Llama, llama_get_embeddings
from llama_cpp.llama_chat_format import Llava15ChatHandler
import time
import numpy as np
import torch
import os
import cv2

import llama_cpp.llava_cpp as llava_cpp
import ctypes
import array

SYSTEM_PROMPT = "A chat between a human and an artifical intellegence assistant. The assistant gives helpful and detailed answers to the humans questions."

def get_llava_image_embedding(clip_ctx, llm, image_bytes):
    prompt = "f{SYSTEM_PROMPT} USER: Describe the following image in detail:" 
    llm.eval(llm.tokenize(prompt.encode("utf8"), add_bos=True))
    data_array = array.array("B", image_bytes)
    c_ubyte_ptr = (
        ctypes.c_ubyte * len(data_array)
    ).from_buffer(data_array)
    embed = (
        llava_cpp.llava_image_embed_make_with_bytes(
            ctx_clip=clip_ctx,
            n_threads=llm.context_params.n_threads,
            image_bytes=c_ubyte_ptr,
            image_bytes_length=len(image_bytes),
        )
    )
    n_past = ctypes.c_int(llm.n_tokens)
    n_past_p = ctypes.pointer(n_past)
    llava_cpp.llava_eval_image_embed(
        ctx_llama=llm.ctx,
        embed=embed,
        n_batch=llm.n_batch,
        n_past=n_past_p,
    )
    assert llm.n_ctx() >= n_past.value
    llm.n_tokens = n_past.value
    llava_cpp.llava_image_embed_free(embed)
    assert llm.n_ctx() >= llm.n_tokens
    return torch.tensor(llama_get_embeddings(llm._ctx.ctx)[:llm.n_embd()]).to("cuda")

def get_llava_text_embedding(llm, text):
    prompt = f"{SYSTEM_PROMPT} USER: Describe the following text in detail:" 
    llm.eval(llm.tokenize(f"{prompt} {text}.".encode("utf8"), add_bos=False))
    embed = torch.tensor(llama_get_embeddings(llm._ctx.ctx)[:llm.n_embd()]).to("cuda")
    return embed

def read_video_frames(video_path, desired_fps):
    """
    Reads the video frame rate and every k-th frame from a video file.

    Args:
    video_path (str): Path to the video file.
    k (int): Interval for selecting frames (every k-th frame is selected).

    Returns:
    tuple: A tuple containing the frame rate and a list of selected frames as bytes.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    k = int(round(fps / desired_fps))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % k == 0:
            # Convert the frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield frame_bytes, frame_count

        frame_count += 1

    cap.release()


if __name__ == "__main__":

    video_dir_path = "/home/ubuntu/csc2508/MSR-VTT/ValVideo/"
    embedding_dir_path = "/home/ubuntu/csc2508/MSR-VTT/llava_13b_embeddings/"
    desired_fps = 1

    if not os.path.exists(embedding_dir_path):
        os.mkdir(embedding_dir_path)

    video_files = os.listdir(video_dir_path)
    video_file_names = [os.path.splitext(v)[0] for v in video_files]

    existing_embedding_names = [os.path.splitext(v)[0] for v in os.listdir(embedding_dir_path)]

    remaining_video_names = set(video_file_names).difference(set(existing_embedding_names))
    print(f"Processing {len(remaining_video_names)} videos")
    print(f"Skipping {len(existing_embedding_names)} videos because they already have embeddings")

    clip_model_path = "/home/ubuntu/csc2508/models/llava-13b/mmproj-model-f16.gguf" 
    llava_model_path = "/home/ubuntu/csc2508/models/llava-13b/ggml-model-q5_k.gguf"
    llava = Llama(model_path=llava_model_path, n_ctx=2048, n_gpu_layers=-1, logits_all=True, embedding=True)
    clip_ctx = llava_cpp.clip_model_load(
        clip_model_path.encode(), 0
    )

    for j, video_name in enumerate(remaining_video_names):
        print(f"[{j}/{len(remaining_video_names)}]")

        video_path = os.path.join(video_dir_path, video_name + ".mp4")
        embedding_path = os.path.join(embedding_dir_path, video_name + ".npz")

        embedding_dict = {}

        s = time.time()
        for frame_bytes, frame_count in read_video_frames(video_path, desired_fps):
            llava.reset()
            # flat cuda torch tensor
            img_embedding = get_llava_image_embedding(clip_ctx, llava, frame_bytes)
            embedding_dict[str(frame_count)] = img_embedding.cpu().numpy()

        duration = time.time() - s 
        print(f"Saving {video_name}'s {len(embedding_dict)} embeddings.")
        print(f"Average per-frame processing time: {duration / len(embedding_dict)}")

        np.savez(embedding_path, **embedding_dict)

