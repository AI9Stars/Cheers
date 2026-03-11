from PIL import Image
from scipy import special
import torch
import numpy as np
from math import e
from param import output
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

class UMMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        if getattr(tokenizer, "image_token_id", None):
            self.image_token_id = tokenizer.image_token_id
        else:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            self.image_token_id = -200
        
        self.image_gen_token = "<image_gen>" if not hasattr(tokenizer, "image_gen_token") else tokenizer.image_gen_token
        if getattr(tokenizer, "image_gen_token_id", None):
            self.image_gen_token_id = tokenizer.image_gen_token_id
        else:
            tokenizer.add_tokens(["<image_gen>"], special_tokens=True)
            self.image_gen_token_id = -300
        
        self.image_gen_start_token = "<im_start>" if not hasattr(tokenizer, "image_gen_start") else tokenizer.image_gen_start
        if getattr(tokenizer, "image_gen_start_token_id", None):
            self.image_gen_start_token_id = tokenizer.image_gen_start_token_id
        else:
            tokenizer.add_tokens(["<im_start>"], special_tokens=True)
            self.image_gen_start_token_id = tokenizer.convert_tokens_to_ids(self.image_gen_start_token)

        self.image_gen_end_token = "<im_end>" if not hasattr(tokenizer, "image_gen_end") else tokenizer.image_gen_end
        if getattr(tokenizer, "image_gen_end_token_id", None):
            self.image_gen_end_token_id = tokenizer.image_gen_end_token_id
        else:
            tokenizer.add_tokens(["<im_end>"], special_tokens=True)
            self.image_gen_end_token_id = tokenizer.convert_tokens_to_ids(self.image_gen_end_token)
        
        self.no_mean_token = "<no_mean>" if not hasattr(tokenizer, "no_mean") else tokenizer.no_mean
        if getattr(tokenizer, "no_mean_id", None):
            self.no_mean_token_id = tokenizer.no_mean_id
        else:
            tokenizer.add_tokens(["<no_mean>"], special_tokens=True)
            self.no_mean_token_id = tokenizer.convert_tokens_to_ids(self.no_mean_token)
            
        if chat_template is None and hasattr(tokenizer, "chat_template"):
            chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
    
    def __call__(self, images=None, text=None, max_resolution=None, add_im_start_id=False, **kwargs):
        if "padding" not in kwargs:
            kwargs["padding"] = True
        if "truncation" not in kwargs:
            kwargs["truncation"] = True
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        return_tensors = kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **kwargs, return_tensors=return_tensors)
        img_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        img_gen_token_id = self.tokenizer.convert_tokens_to_ids(self.image_gen_token)
        if add_im_start_id:
            B, T = text_inputs["input_ids"].shape
            new_input_ids = torch.full((B, T+1), self.tokenizer.pad_token_id)
            new_input_ids[:, :T] = text_inputs["input_ids"]
            is_valid = (text_inputs["input_ids"] != self.tokenizer.pad_token_id)
            valid_len = is_valid.sum(dim=1)
        else:
            new_input_ids = text_inputs["input_ids"]

        t = []
        und_gen_mask_list = []
        for i, ids in enumerate(text_inputs["input_ids"]):
            for j, token_id in enumerate(ids):
                if token_id == img_token_id:
                    new_input_ids[i][j] = self.image_token_id
                    t.append(torch.tensor([1.0]))
                    und_gen_mask_list.append(1)
                elif token_id == img_gen_token_id:
                    new_input_ids[i][j] = self.image_gen_token_id
                    t.append(torch.rand(1))
                    und_gen_mask_list.append(0)
        
        image_inputs = {}
        pixel_values, grid_hws = [], []
        if images is not None:
            image_idx = 0
            for per_images in images if isinstance(images, list) else [images]:
                if per_images is None:
                    dummy_image = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
                    image_info = self.image_processor(images=dummy_image)
                else:
                    for per_image in per_images if isinstance(per_images, list) else[per_images]:
                        if und_gen_mask_list[image_idx] == 0:
                            image_info = self.image_processor(images=per_image, max_resolution=max_resolution, und=False)
                        else:
                            image_info = self.image_processor(images=per_image, max_resolution=max_resolution)
                        image_idx += 1
                pixel_values.append(image_info.pixel_values)
                grid_hws.append(image_info.grid_hws)
            pixel_values = torch.concat(pixel_values, dim=0)
            grid_hws = torch.concat(grid_hws, dim=0)
            image_inputs.update({'pixel_values': pixel_values, 'grid_hws': grid_hws})

        if len(t) > 0:
            t = torch.cat(t)
            image_inputs.update({"t":t})
        if add_im_start_id:
            for b in range(B):
                pos = valid_len[b].item()
                new_input_ids[b, pos] = self.image_gen_start_token_id
            attention_mask = torch.cat([
                text_inputs["attention_mask"],
                (new_input_ids[:, -1] != self.tokenizer.pad_token_id).long().unsqueeze(1)
            ], dim=1)
            text_inputs["attention_mask"] = attention_mask
        text_inputs["input_ids"] = new_input_ids
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

__all__ = ["UMMProcessor"]