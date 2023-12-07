import json
from pathlib import Path
import os

msr_path = Path("/home/bagro/MSR-VTT") 
with open(str(Path(msr_path / "train_val_annotation/train_val_videodatainfo.json")), "r") as f:
    data = json.load(f)



# new_json = {
#     "videos": []
#     "sentences": []
# }

validation_videos = []
for video_data in data["videos"]:
    if video_data["split"] != "validate":
        video_id = video_data["video_id"] + ".mp4"
        if os.path.exists(str(msr_path / "TrainValVideo" / video_id)):
            os.remove(str(msr_path / "TrainValVideo" / video_id))
    else:
        video_id = video_data["video_id"]
        # new_json["videos"].append(video_data)
        validation_videos.append(video_id)

validation_videos = set(validation_videos)

video_to_sentence_mapping = {}
sentence_to_video_mapping = {}

for sentence_data in data["sentences"]:
    video_id = sentence_data["video_id"]
    if video_id not in validation_videos:
        continue
    if video_id not in video_to_sentence_mapping:
        video_to_sentence_mapping[video_id] = []

    sentence = sentence_data["caption"] 
    video_to_sentence_mapping[video_id].append(sentence)

    if sentence not in sentence_to_video_mapping:
        sentence_to_video_mapping[sentence] = []
    sentence_to_video_mapping[sentence].append(video_id)

msr_path = Path("/home/bagro/MSR-VTT") 
new_json = {
    "video_to_sentence_mapping": video_to_sentence_mapping,
    "sentence_to_video_mapping": sentence_to_video_mapping,
}
with open(str(Path(msr_path / "train_val_annotation/val_info.json")), "w") as f:
    json.dump(new_json, f)