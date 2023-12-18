import json
from pathlib import Path
import os

msr_path = Path("/home/ubuntu/") 
with open(str(Path(msr_path / "test_videodatainfo.json")), "r") as f:
    data = json.load(f)


with open(str(Path(msr_path / "val_list_jsfusion.txt")), "r") as f:
    validation_videos = set(str(f.read()).split("\n"))

for video in validation_videos:
    path = msr_path / "TestVideo" / (video + ".mp4")
    assert os.path.exists(str(path)), str(path)

new_json = {
    "videos": [],
    "sentences": [],
}

for video_data in data["videos"]:
    video_id = video_data["video_id"] # + ".mp4"
    if video_id not in validation_videos:
        if os.path.exists(str(msr_path / "TestVideo" / (video_id + ".mp4"))):
            os.remove(str(msr_path / "TestVideo" / (video_id + ".mp4")))

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

new_json = {
    "video_to_sentence_mapping": video_to_sentence_mapping,
    "sentence_to_video_mapping": sentence_to_video_mapping,
}
with open(str(Path(msr_path / "test_info.json")), "w") as f:
    json.dump(new_json, f)