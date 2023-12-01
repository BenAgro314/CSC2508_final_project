import os
import json
from pathlib import Path
import subprocess

from video_retrieval_models.common import AudioToTextProtocol


class Whisper(AudioToTextProtocol):

    def __init__(self):
        whisper_dir = Path(__file__).parent.parent.parent / "whisper.cpp"
        self.whisper_path = str(whisper_dir / "main")
        self.model_path = os.path.join(str(whisper_dir), "models", "ggml-base.en.bin") 

    def call_whisper(self, audio_file_path: str, output_file_path: str) -> None:
        subprocess.run(f"{self.whisper_path} -f {audio_file_path} --output-json --output-file {output_file_path} --model {self.model_path}", shell=True)

    def build_index(self, audio_dir_path: str) -> None:
        audio_files = os.listdir(audio_dir_path)

        out_dir = str(Path(Path(audio_dir_path).parent / "whisper_caption_files"))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        count = 0
        for audio_filename in audio_files: 
            if not audio_filename.endswith(".wav"):
                continue

            count += 1

            output_file_path = os.path.join(out_dir, os.path.splitext(audio_filename)[0])
            audio_file_path = os.path.join(audio_dir_path, audio_filename)
            if os.path.exists(output_file_path + ".json"):
                print(f"Skipping {audio_filename}, its json already exists")
                continue

            print(f"Calling whisper on {audio_file_path}")
            self.call_whisper(audio_file_path, output_file_path)

        if count == 0:
            print(f"WARNING: No valid .wav files in the audio folder {audio_dir_path}, found {audio_files}")


if __name__ == "__main__":
    model = Whisper()
    model.build_index("/home/bagro/audios/")