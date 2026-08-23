import os
import torch
from torchvision.utils import save_image
from transformers import AutoModelForCausalLM, AutoProcessor
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ckpt = "Cheers-CKPT/v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
model.to(device)
model = model.to(torch.bfloat16)
model.eval()

content = """
In the center of a bustling intersection, a large tree with a thick trunk and sprawling branches stands out amidst the concrete. 
Its green leaves contrast sharply with the grey asphalt roads that converge around it. Traffic lights and street signs are positioned awkwardly around the tree's base, creating an unusual juxtaposition of nature and urban infrastructure.
"""
images_batch = [None]

messages_batch = [
        [{"role": "user", "content": content}],
    ]

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
inputs = processor(text=texts, images=images_batch, return_tensors="pt", add_im_start_id=True)
inputs = {k: (v.to(device=device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

gen_config = {
    "max_length": 300,
    "cfg_scale": 9.5,
    "temperature": 0.0,
    "num_inference_steps": 80,
    "alpha": 0.5,
    "edit_image": False,
}

inputs.update(gen_config)
generated = model.generate(**inputs)
input_ids = generated["input_ids"]
images = generated["images"][0]

current_img = images[0]
current_img = current_img.clamp(0.0, 1.0)
save_image(current_img, f"outputs/case_.png")

print(f"Save image: outputs/case_.png")