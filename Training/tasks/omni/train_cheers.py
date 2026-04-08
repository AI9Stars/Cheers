import json
import os
import random
import time
import numpy as np
import yaml
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import wandb
from PIL import Image, ImageFile
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_interleave_dataset,
    build_iterative_dataset,
    build_mapping_dataset,
    build_multimodal_chat_template,
)
from veomni.data.constants import IMAGE_INPUT_INDEX
from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, save_model_assets, save_model_weights, set_trainable_modules
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from safetensors.torch import load_file, safe_open
from transformers import AutoTokenizer

from collections import namedtuple

ImageFile.LOAD_TRUNCATED_IMAGES = True
LPIPS_CKPT_PATH = "local vgg.pth path" # if use

class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor: torch.Tensor):
        h = self.slice1(tensor)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(tensor ** 2, dim=1, keepdim=True))
    return tensor / (norm_factor + eps)


def _spatial_average(tensor: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return tensor.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    """Learned perceptual metric used by VQGAN."""

    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_pretrained_weights(LPIPS_CKPT_PATH)
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, ckpt: str = "vgg_lpips") -> None:
        state = torch.load(ckpt, map_location=torch.device("cpu"))
        self.load_state_dict(state, strict=False)
        print(f"[LPIPS] Loaded pretrained weights from {ckpt}")

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        input_scaled, target_scaled = self.scaling_layer(input), self.scaling_layer(target)
        feats_input = self.net(input_scaled)
        feats_target = self.net(target_scaled)
        diffs = []
        lin_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for idx, (feat_in, feat_tgt) in enumerate(zip(feats_input, feats_target)):
            feat_in = _normalize(feat_in)
            feat_tgt = _normalize(feat_tgt)
            diff = (feat_in - feat_tgt) ** 2
            diffs.append(_spatial_average(lin_layers[idx].model(diff), keepdim=True))

        value = diffs[0]
        for diff in diffs[1:]:
            value = value + diff

        if reduction == "none":
            return value
        if reduction == "sum":
            return torch.sum(value)
        if reduction == "mean":
            return torch.mean(value)
        raise ValueError(f"Unsupported reduction '{reduction}'")

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate

logger = helper.create_logger(__name__)

def load_model_weights_auto(model_dir, device):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        logger.info_rank0(f"Detected sharded safetensors at {index_path}")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        shard_paths = sorted(list(set(weight_map.values())))
        state_dict = {}
        for shard_name in shard_paths:
            shard_path = os.path.join(model_dir, shard_name)
            logger.info_rank0(f"Loading {shard_path}")
            with safe_open(shard_path, framework="pt", device=device) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        logger.info_rank0(f"loaded {len(state_dict)} tensors from {len(shard_paths)} shards.")
    else:
        safetensor_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
            logger.info_rank0(f"Loading weights from {safetensor_path} (safetensors)")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            logger.info_rank0(f"Loading weights from {bin_path} (torch.bin)")
    return state_dict

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]

def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    image_folder,
    cfg_data_rate = None,
    **kwargs,
):
    """
    Processes multimodal example with qwen2vl's pre-processor.
    """
    conversations = sample["conversations"]  # text-only data
    conversations = conv_preprocess('custom_data', conversations, **kwargs)
    tokenized_example = chat_template.encode_messages(conversations)
    input_ids = tokenized_example["input_ids"]
    new_input_ids = None
    if cfg_data_rate is not None:
        image_token = processor.image_token_id
        image_gen_token_id = processor.image_gen_token_id
        image_gen_start_token_id = processor.image_gen_start_token_id
        image_gen_end_token_id = processor.image_gen_end_token_id
        no_mean_token_id = processor.no_mean_token_id
        # allowed_ids = [image_gen_token_id, image_gen_start_token_id, image_gen_end_token_id]
        allowed_ids = [image_gen_token_id, image_gen_start_token_id, image_gen_end_token_id, image_token]
        if random.random() < cfg_data_rate and image_gen_token_id in input_ids:
            new_input_ids = [
                token if token in allowed_ids else no_mean_token_id
                for token in input_ids
            ]
    t = []
    und_gen_mask_list = []
    use_cfg_data = False
    for token_id in tokenized_example["input_ids"]:
        if token_id == processor.image_gen_token_id:
            use_cfg_data = True
            eps = 1e-6
            t.append(torch.rand(1).clamp(eps, 1-eps))
            und_gen_mask_list.append(0)
        elif token_id == processor.image_token_id:
            t.append(torch.tensor([1.0]))
            und_gen_mask_list.append(1)

    if use_cfg_data and new_input_ids is not None: 
        tokenized_example["input_ids"] = new_input_ids
    if len(t) == 0:
        t.append(torch.tensor([0.0]))
    t = torch.cat(t, dim=0)
    tokenized_example["t"] = t

    image_idx = 0
    pixel_values, grid_hws = [], []
    if "image" in sample and sample["image"] and sample["image"] != "null":
        if isinstance(sample["image"], list):
            images_path = sample["image"]
        else:
            sample["image"] = sample["image"].replace(';', ',')
            images_path = sample["image"].split(',')
        for image_path in images_path:
            image_path = image_path.strip()
            image_path = os.path.join(image_folder, image_path)
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                except:
                    print(f"=============={image_path}=================")
                    image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            else:
                image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            if image_idx < len(und_gen_mask_list):
                if und_gen_mask_list[image_idx] == 0:
                    image_info = processor.image_processor(images=image, und=False)
                else:
                    image_info = processor.image_processor(images=image, und=True)
            else:
                image_info = processor.image_processor(images=image, und=True)
            image_idx += 1
            pixel_values.append(image_info.pixel_values)
            grid_hws.append(image_info.grid_hws)
    else:
        dummy_image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        image_info = processor.image_processor(images=dummy_image)
        pixel_values.append(image_info.pixel_values)
        grid_hws.append(image_info.grid_hws)

    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    tokenized_example["pixel_values"] = torch.concat(pixel_values, dim=0).to(torch.bfloat16)
    tokenized_example["grid_hws"] = torch.concat(grid_hws, dim=0)
    return [tokenized_example]

@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable_modules: List[str] = field(
        default=None,
        metadata={"help": "Choice the train modules. Default is all."}
    )
    vit_lr: float = field(
        default=None,
    )
    cfg_data_rate: float = field(
        default=None
    )
    und_only: bool = field(
        default=False
    )
    use_l1_loss: bool = field(
        default=False
    )
    use_lpips_loss: bool = field(
        default=False
    )
    use_dist_loss: bool = field(
        default=False
    )

@dataclass
class MyModelArguments(ModelArguments):
    torch_dtype: str = field(
        default=None,
        metadata={"help": "model.to(dtype)"},
    )
    model_weights_path: str = field(
        default=None,
    )
    vae_weights: str = field(
        default=None,
    )
    siglip2_weights: str = field(
        default=None,
    )
    language_model_weights_path: str = field(
        default=None,
    )

@dataclass
class MyDataArguments(DataArguments):
    image_folder: str = field(
        default=None,
        metadata={"help": "Used for dataloader"},
    )

@dataclass
class Arguments:
    model: "MyModelArguments" = field(default_factory=MyModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)

def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_nccl_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)
    
    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )
    
    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        torch_dtype="bfloat16",
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
        attn_implementation="sdpa",
    )

    #### 在GPU上加载模型的方式 ####
    vlm_state = model.state_dict()
    if args.model.model_weights_path is not None:
        model_weights = load_model_weights_auto(args.model.model_weights_path, device="cpu")
        for k, v in model_weights.items():
            if k in vlm_state:
                vlm_state[k].copy_(v)
            else:
                logger.info_rank0(f"No {v} in vlm.")
        logger.info_rank0("Sucessful load weights from model_weights")
        del model_weights

    if args.model.vae_weights is not None:
        vae_weights = load_model_weights_auto(args.model.vae_weights, device="cpu")
        tgt_key_used = False
        tgt_key_decoder_used = False
        for k, v in vae_weights.items():
            tgt_key = f"model.vae_model.{k}"
            tgt_key_decoder = f"model.vae_decoder_projector.{k}"
            if tgt_key in vlm_state:
                vlm_state[tgt_key].copy_(v)
                tgt_key_used = True
            else:
                import pdb; pdb.set_trace()
                logger.info_rank0(f"No {tgt_key} in vlm")
            if tgt_key_decoder in vlm_state:
                vlm_state[tgt_key_decoder].copy_(v)
                tgt_key_decoder_used = True
        if tgt_key_decoder_used and tgt_key_used:
            logger.info_rank0("Sucessful load weights from vae_model")
        else:
            logger.info_rank0("Some weights from vae_model is not used")
        del vae_weights

    if args.model.siglip2_weights is not None:
        siglip2_weights = load_model_weights_auto(args.model.siglip2_weights, device="cpu")
        for k, v in siglip2_weights.items():
            if "vision_model" in k:
                k = k.replace('vision_model', "vision_representation")
                tgt_key = f"model.{k}"
                if tgt_key in vlm_state:
                    vlm_state[tgt_key].copy_(v)
                else:
                    logger.info_rank0(f"No {tgt_key} in vlm")

    if args.model.language_model_weights_path is not None:
        language_model_weights = load_model_weights_auto(args.model.language_model_weights_path, device="cpu")
        language_embed_tokens_weight = language_model_weights["model.embed_tokens.weight"]
        # lm_weight = language_model_weights["lm_head.weight"]
        for k, v in language_model_weights.items():
            tgt_key = k.replace('model', 'model.language_model')
            if tgt_key in vlm_state and not tgt_key.startswith("model.language_model.embed_tokens"):
                vlm_state[tgt_key].copy_(v)
            elif tgt_key.startswith("model.language_model.embed_tokens"):
                pass
            else:
                logger.info_rank0(f"No {tgt_key} in vlm.")
        vlm_state["model.language_model.embed_tokens.weight"][: language_embed_tokens_weight.shape[0]].copy_(language_embed_tokens_weight)
        logger.info_rank0(f"Copied embeddings from language model to vlm")
        logger.info_rank0("Sucessful load weights from language_model")
        del language_model_weights
        del language_embed_tokens_weight
        # del lm_weight
    helper.empty_cache()
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)
    transform = partial(
        process_sample,
        processor=processor,
        chat_template=chat_template,
        cfg_data_rate=args.train.cfg_data_rate,
        image_folder=args.data.image_folder
    )

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.train.rmpad_with_pos_ids:
        data_collate_fn.append(OmniDataCollatorWithPacking())
    else:
        data_collate_fn.append(OmniDataCollatorWithPadding())

    if os.path.isfile(args.data.train_path) and args.data.train_path.lower().endswith((".yaml", "yml")):
        with open(args.data.train_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        json_paths = []
        if isinstance(data, dict) and "datasets" in data:
            for item in data["datasets"]:
                if isinstance(item, dict) and "json_path" in item:
                    json_paths.append(item["json_path"].strip(", "))
        args.data.train_path = ",".join(json_paths)
        logger.info_rank0(args.data.train_path)

    if args.data.dataloader_type == "native":
        if args.data.enable_multisource:
            logger.info_rank0("Start building interleave dataset")
            train_dataset = build_interleave_dataset(
                args.data.train_path, args.data.datasets_type, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(
                args.data.train_path, transform=transform, seed=args.train.seed, source_name=args.data.source_name
            )
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(
                args.data.train_path, transform=transform, source_name=args.data.source_name
            )

        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    model = set_trainable_modules(model, args.train.trainable_modules)
    if args.train.data_parallel_mode == "fsdp1":
        fsdp_kwargs["use_orig_params"] = True
    elif args.train.data_parallel_mode == "fsdp2":
        fsdp_kwargs["use_orig_params"] = True

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    if args.train.vit_lr is not None:
        optimizer = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=False,
            optimizer_type=args.train.optimizer,
            param_groups=get_param_groups(model, args.train.lr, args.train.vit_lr)
        )
    else:
        optimizer = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=False,
            optimizer_type=args.train.optimizer,
        )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )
    
    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    if args.model.torch_dtype is not None:
        if args.model.torch_dtype == 'bfloat16':
            model.to(torch.bfloat16)
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    
    if args.train.use_lpips_loss:
        lpips = LPIPS().to(model.device)
        lpips.eval()

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            total_text_loss = 0
            total_image_loss = 0
            synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)
                    image_loss = model_outputs.image_loss   
                    text_loss = model_outputs.text_loss
                    if not args.train.und_only:
                        alpha = 1.0
                        loss = image_loss + alpha * text_loss
                    else:
                        loss = text_loss
                        image_loss = 0 * image_loss
                    loss = loss / len(micro_batches)
                    text_loss = text_loss / len(micro_batch)
                    image_loss = image_loss / len(micro_batch)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                total_text_loss += text_loss.item()
                total_image_loss += image_loss.item()
                del micro_batch
            # import pdb; pdb.set_trace()
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(1.0).item()
            elif args.train.data_parallel_mode == "fsdp2":
                grad_norm = model.clip_grad_norm_(1.0).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            show_loss_part = f"loss: {total_loss:.4f}, "
            if image_loss is not None:
                show_loss_part += f"i_loss: {total_image_loss:.2f}, "
            if text_loss is not None:
                show_loss_part += f"t_loss: {total_text_loss:.2f}, "

            data_loader_tqdm.set_postfix_str(show_loss_part + f"grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()
                    helper.upload_trace(args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir)

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0:
        if args.train.save_hf_weights and save_checkpoint_path is not None:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager)
            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()