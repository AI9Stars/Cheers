import os
import json
import argparse
import torch

from tqdm import tqdm
from torchvision.utils import save_image
from transformers import AutoModelForCausalLM, AutoProcessor

import torch.multiprocessing as mp
from torchvision.transforms import ToPILImage
import numpy as np
from .base_eval_model import Base_eval_model
from PIL import Image
import torch.distributed as dist

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def tensor_to_pil(tensor_imgs):
    to_pil = ToPILImage()
    pil_images = []
    for img_tensor in tensor_imgs:
        pil_images.append(to_pil(img_tensor))
    return pil_images

class Cheers_eval(Base_eval_model):

    def __init__(self, model_path=None, batch_size=1, cfg=None, steps=None, resolution=512, alpha=None, **kwargs):
        self.batch_size = batch_size
        self.model_path = model_path
        self.cfg = cfg
        self.steps = steps
        self.resolution = resolution
        self.alpha = alpha
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        if os.environ["WORLD_SIZE"] == "1":
            self.device = "cuda"
        else:
            rank = dist.get_rank()
            rank = 0
            self.device = f"cuda:{rank}"
        print(self.device)
        self.model = model.to(self.device).to(torch.bfloat16)
        self.model.eval()
        

    def gen_t2i(self, prompt, img_num, dataset_name):
        rank, world_size = get_rank_and_world_size()
        image_list = []
        messages_batch = [[{"role": "user", "content": prompt}] for _ in range(img_num)]
        images_batch = [None for _ in range(img_num)]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        inputs = self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            add_im_start_id=True,
        )

        inputs = {k: (v.to(device=self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()}

        gen_config = {
            "max_length": 500,
            "cfg_scale": self.cfg,
            "temperature": 0.0,
            "num_inference_steps": self.steps,
            "alpha": self.alpha,
        }

        print(f'gen_config {gen_config}')

        inputs.update(gen_config)

        with torch.no_grad():
            # breakpoint()
            generated = self.model.generate(**inputs)

        images = generated["images"][0]

        # breakpoint()
        image_list = []
        for image_num, current_img in enumerate(images):
            # breakpoint()
            current_img = current_img.to(dtype=torch.float)
            # current_img = current_img.clamp(0.0, 1.0)
            current_img = current_img.to("cpu").permute(1,2,0).numpy()
            current_img = np.clip(current_img * 255.0, 0, 255).astype(np.uint8)
            # current_img = current_img[:, :, ::-1]
            current_img = Image.fromarray(current_img)
            image_list.append(current_img) # image_list.append(tensor_to_pil(current_img))


        return image_list

    def gen_i2i(self, prompt, input_images, img_num):
        pass

    def bulid_prompt(self, prompt):
        pass 