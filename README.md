# Final Project for CSC2508: Zero-Shot Video Retrieval With Vision Language Models

## Google Colab Demo

You can find a google colab demo here: https://colab.research.google.com/drive/1yzXtIWKpKGXke4TqUAGQMtrVbSj_PRsh?usp=sharing,
which you can use to run video retrieval on your data.

## System Requirements

- Ubuntu 20.04
- Nvidia GPU with at least 14GB of memory
- CUDA Version >= 12.0
- Python version >= 3.8

## Setup

Clone repo and submodules
```bash
git clone --recurse-submodules https://github.com/BenAgro314/CSC2508_final_project.git
```

Build `llama.cpp`:
```bash
cd CSC2508_final_project/llama.cpp
make LLAMA_CUBLAS=1
```

Build `whisper.cpp`:
```bash
cd CSC2508_final_project/whisper.cpp
bash ./models/download-ggml-model.sh base.en
WHISPER_CUBLAS=1 make -j
```

Install python dependencies

```bash
pip install -r requirements.txt
```

## Building the Index For Video Retrieval

Assume the videos are stored in `/videos/*.mp4`.
First generate `.wav` files for each `.mp4`, so whisper can process them:
```bash
bash ./whisper.cpp/mp4s_to_wavs.sh /videos /audios
```

OPTIONAL: Run whisper to generate closed captions (in python)
```python
from video_retrieval_models.audio_models.whisper import Whisper
whisper = Whisper()
whisper.build_index("/audios")
```
(this will populate `/audios` with `.json` files containing the closed captions)


Now we can run LLaVA.
Make an empty directory for the documents that will be created, e.g., `mkdir /docs`.
Also, download model files for LLaVA (e.g., from https://huggingface.co/mys/ggml_llava-v1.5-13b/tree/main).
Then you can run the command:
```bash
./llama.cpp/llava-test -m /models/ggml_llava-v1.5-13b.q5_k.gguf --mmproj /models/llava/mmproj-model-f16.gguf --video-dir /videos --doc-dir /docs --audio-captions-dir /audio -ngl 64 --temp 0.1
```
Note that this will take a while, depending on how many videos you have and the hardware you are running it on. On my machine it takes about 4s per frame, and we process at 1 frame per second,
so roughly 4x realtime.
This will create a set of documents in the form of `.json` files, one for each video, in `/docs`.

## Text Retrieval

Here is a code snippet for example text retrieval using `AnglE` embeddings:
```python
from video_retrieval_models.text_models.angle_embeddings import AngleEmbeddings
device = "cuda"
pool = "max" # indicates we are using the `max` over the per-caption scores. You can also use `mean` (worse performance)
text_model = AngleEmbeddings(device, pool=pool)
doc_dir = "/docs"
text_model.build_index(doc_dir)
# change these parameters as you please
queries: List[str] = # insert your sentences here
topk = 10
for query in queries:
    docs = text_model.retrieve(sentence, topk=topk) # list of `topk` document names
```
Other retrieval options include
`BM25ModelDocRetrieval`, `DocLevelAngleEmbeddings` (using `mean` or `max` pooling on the per-caption embeddings
to generate a document-level embedding), and `DocLevelSentenceTransformerModel` .