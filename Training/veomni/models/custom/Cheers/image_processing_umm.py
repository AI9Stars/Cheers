from networkx import to_numpy_array
import numpy as np
import torch
from PIL import Image, ImageOps
import math
from functools import partial, reduce
from transformers.image_transforms import (
    convert_to_rgb,
    center_crop,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_utils import ImageInput, ChannelDimension, PILImageResampling, to_numpy_array

class UMMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "grid_hws"]
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(256, 256),
        crop_size = None, 
        resample=PILImageResampling.BICUBIC, 
        rescale_factor=1 / 255, 
        data_format=ChannelDimension.FIRST,
        scale_resolution=256,
        patch_size=16, 
        **kwargs,
    ):
        super().__init__(**kwargs)
        crop_size = crop_size if crop_size is not None else {"height": 256, "width": 256}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size

    def preprocess(self, image, max_resolution=None, return_tensors = 'pt', und=True, **kwargs) -> BatchFeature:
        if max_resolution is not None:
            scale_resolution = max_resolution
        else:
            scale_resolution = self.scale_resolution
        if image is not None:
            pixel_values, grid_hws = [], []
            if und:
                image = self._preprocess_und(image, scale_resolution)
            else:
                image = self._preprocess_gen(image, scale_resolution)
            if not torch.is_tensor(image):
                image = torch.tensor(image)
            _,H,W = image.shape
            grid_h = int(H // self.patch_size)
            grid_w = int(W // self.patch_size)
            grid_hw = (grid_h, grid_w)
            pixel_values = torch.stack([image], dim=0)
            grid_hws = torch.tensor([grid_hw])
            data = {
                "pixel_values": pixel_values,
                "grid_hws": grid_hws
            }
        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def _preprocess_gen(self, source_image, scale_resolution):
        w, h = source_image.size
        scale = scale_resolution / min(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        source_image = source_image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        source_image = [source_image]
        transforms = [
            convert_to_rgb,
            to_numpy_array,
        ]
        transforms.append(partial(center_crop, size=(scale_resolution, scale_resolution)))
        transforms.append(partial(rescale, scale=self.rescale_factor, data_format=self.data_format))
        transforms.append(partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format))
        image = reduce(lambda x, f: [*map(f, x)], transforms, source_image)
        return image[0] if len(image) == 1 else image

    def _preprocess_und(self, source_image, scale_resolution):
        w, h = source_image.size
        scale = min(scale_resolution / h, scale_resolution / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized_image = source_image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        pad_w = scale_resolution - new_w
        pad_h = scale_resolution - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        new_image = ImageOps.expand(resized_image, border=(left, top, right, bottom), fill=(0,0,0))
        # new_image.save("test_path")
        source_image = [new_image]
        transforms = [
            convert_to_rgb,
            to_numpy_array
        ]
        transforms.append(partial(rescale, scale=self.rescale_factor, data_format=self.data_format))
        transforms.append(partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format))
        image = reduce(lambda x, f: [*map(f, x)], transforms, source_image)
        return image[0] if len(image) == 1 else image

__all__ = ["UMMImageProcessor"]

