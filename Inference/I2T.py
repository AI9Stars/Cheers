import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ckpt = "Cheers-CKPT/v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
model = model.to(torch.bfloat16)
model.to(device)
model.eval()

content = "<im_start><image><im_end>\n Discribe this image."

img = Image.open("fig/logo.png")
images_batch = [img,]

messages_batch = [
        [{"role": "user", "content": content}],
    ]

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
inputs = processor(text=texts, images=images_batch, return_tensors="pt", add_im_start_id=False)
inputs = {k: (v.to(device=device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

gen_config = {
    "max_length": 150,
    "temperature": 0.3,
}

inputs.update(gen_config)
generated = model.generate(**inputs)
input_ids = generated["input_ids"]

print(processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True))