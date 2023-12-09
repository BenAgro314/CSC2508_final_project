import json
from pathlib import Path
import os
import sys

# 1. Load Labels
msr_path = Path("/home/bagro/MSR-VTT") 
with open(str(Path(msr_path / "train_val_annotation/val_info.json")), "r") as f:
    val_info = json.load(f)

# 2. load predictions
assert len(sys.argv) == 2
preds_path = Path(sys.argv[1]) 
with open(str(preds_path), "r") as f:
    preds = json.load(f)

# preds format:
# {
#   <sentence>: [list of length 10 with best videos]
# }

sentence_to_video_mapping = val_info["sentence_to_video_mapping"]
print(f"Running validation on {len(sentence_to_video_mapping)} sentences")

rs = [1, 5, 10, 20, 30, 50, 100]

tps = {
    r: 0 for r in rs
}

sentence_count = 0

for i, sentence in enumerate(sentence_to_video_mapping):
    print(f"Sentence: [{i}/{len(sentence_to_video_mapping)}]")
    if sentence not in preds:
        print(f"Warning skipping sentence '{sentence}' because it is not in preds")
        continue
    assert len(preds[sentence]) >= 10, "Must produce top 10 rankings"
    sentence_count += 1
    gt_video_paths = set(sentence_to_video_mapping[sentence])
    for r in rs:
        hit = False
        for v in preds[sentence][:r]:
            if v in gt_video_paths:
                hit = True
                break
        if hit:
            tps[r] += 1
        else:
            print("Failure case:")
            print(sentence, preds[sentence][:r])

recall = {r: (tp/sentence_count) for r, tp in tps.items()} 

print(f"Recall metrics: {recall}")