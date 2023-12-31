from video_retrieval_models.text_models.doc_level_angle_embeddings_no_pool import DocLevelAngleEmbeddingsNoPool
from pathlib import Path
import json

device = "cuda"
# Best so far:
text_model = DocLevelAngleEmbeddingsNoPool(device, pool="mean") 

doc_dir = "/home/ubuntu/csc2508/MSR-VTT/llava_docs_13b/"
text_model.build_index(doc_dir)

# 1. Load Queries
msr_path = Path("/home/ubuntu/csc2508/MSR-VTT") 
with open(str(Path(msr_path / "train_val_annotation/val_info.json")), "r") as f:
    val_info = json.load(f)

video_to_sentence_mapping = val_info["video_to_sentence_mapping"]
sentence_to_video_mapping = val_info["sentence_to_video_mapping"]

preds = {}

topk = 100
for i, video in enumerate(video_to_sentence_mapping):
    # print(f"Processing video [{i}/{len(text_model.corpus)}]")
    if video not in text_model.corpus:
        print(f"Skpping: {video}, it is not in the corpus")
        continue
    for sentence in video_to_sentence_mapping[video]:
        if sentence not in preds:
            preds[sentence] = []
        else:
            continue
        print(f"Trying to retrieve sentence: {sentence}")
        # docs = text_model.retrieve(sentence, topk=topk)
        docs = text_model.retrieve(sentence, topk=topk)
        preds[sentence] += docs
        print(docs)
        # for 
            # doc_name = docs[i][0].name
             # print(doc_name)
            # preds[sentence].append(doc_name)

preds_path = Path("/home/ubuntu/csc2508/MSR-VTT/doc_level_angle_preds_no_pool.json") 
with open(str(preds_path), "w") as f:
    json.dump(preds, f)

print(f"Saved preds to {preds_path}")

