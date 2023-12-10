#%%
from llama_cpp import Llama, llama_get_embeddings
from llama_cpp.llama_chat_format import Llava15ChatHandler
import time
import torch

import llama_cpp.llava_cpp as llava_cpp
import ctypes
import array


#%%
clip_model_path = "/home/ubuntu/csc2508/models/llava-13b/mmproj-model-f16.gguf" 
#chat_handler = Llava15ChatHandler(clip_model_path="/home/ubuntu/csc2508/models/llava-13b/mmproj-model-f16.gguf")
llava = Llama(model_path="/home/ubuntu/csc2508/models/llava-13b/ggml-model-q5_k.gguf", n_ctx=2048, n_gpu_layers=-1, logits_all=True, embedding=True)
clip_ctx = llava_cpp.clip_model_load(
    clip_model_path.encode(), 0
)

#%%

def get_llava_image_embedding(clip_ctx, llm, image_path):
    system_prompt = "Represent this data for searching relevant images:" 
    llm.eval(llm.tokenize(system_prompt.encode("utf8"), add_bos=True))
    with open(image_path, "rb") as f:
        image_bytes = f.read()
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
    llm.eval(llm.tokenize(f"Represent this data for searching relevant images: {text}.".encode("utf8"), add_bos=False))
    embed = torch.tensor(llama_get_embeddings(llm._ctx.ctx)[:llm.n_embd()]).to("cuda")
    return embed

#%%

image_path = "/home/ubuntu/csc2508/Orangutan.jpg"

llava.reset()
img_embedding = get_llava_image_embedding(clip_ctx, llava, image_path)

llava.reset()
pos_embed = get_llava_text_embedding(llava, "A documentary about Borneo")

llava.reset()
neg_embed = get_llava_text_embedding(llava, "A documentary about Toronto")

cosine_sim = torch.nn.CosineSimilarity(dim=0)
print(f"Cosine:")
print(cosine_sim(img_embedding, pos_embed))
print(cosine_sim(img_embedding, neg_embed))

print(f"Dot:")
print(torch.dot(img_embedding, pos_embed))
print(torch.dot(img_embedding, neg_embed))
