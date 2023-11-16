from primitives.caption import Caption
from transformers import AutoTokenizer, BitsAndBytesConfig
import tqdm
from llava.model import LlavaLlamaForCausalLM
from pathlib import Path
from utils.video_to_images import video_to_PIL
from video_retrieval_models.common import VideoToTextProtocol, TextRetrievalProtocol, BaseVideoRetrievalModel
from primitives.document import Document
import torch
import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
import cv2
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer



class LlavaModel(VideoToTextProtocol):

    def __init__(self, device: torch.device):
        model_path = "4bit/llava-v1.5-7b-5GB"
        kwargs = {"device_map": "auto"}
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device='cuda')
        self.image_processor = vision_tower.image_processor

        self.process_fps = 1

    def caption_image(self, image, prompt):
        disable_torch_init()
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        inp = f"{roles[0]}: {prompt}"
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        raw_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                        max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        output = outputs.rsplit('</s>', 1)[0]
        return output

    def build_index(self, video_dir_path: str) -> None:

        video_path = video_dir_path
        doc_path = str(Path(video_dir_path).parent / "llava_documents")
        if not os.path.isdir(doc_path):
            os.mkdir(doc_path)

        videos = {Path(v).stem: v for v in os.listdir(video_path)}
        docs = {Path(d).stem: d for d in os.listdir(doc_path)}

        assert len(docs) <= len(videos), f"Documents {set(docs.keys()).difference(set(videos.keys()))} do not have corresponding videos"
        unprocessed_vids = set(videos.keys()).difference(set(docs.keys()))
        print(f"Unprocessed videos: {unprocessed_vids}")

        for vid_name in unprocessed_vids:
            vid_path = os.path.join(video_path, videos[vid_name])
            print(f"Processing {vid_name}")

            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            document = Document(name=vid_name, video_path=vid_path, fps=fps)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Video fps: {fps}, total frames: {total_frames}")
            subsample_rate = int(round(fps / self.process_fps))

            for frame_number, pil in tqdm.tqdm(video_to_PIL(vid_path, subsample_stride=subsample_rate)):
                query = "Describe the image and color details."
                output = self.caption_image(pil, query)
                document.add_caption(frame=frame_number, caption=Caption(output))

            print(document)
            document.save(doc_path)

        self.doc_path = doc_path

    @property
    def documents_path(self) -> str:
        return self.doc_path
