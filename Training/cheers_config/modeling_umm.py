from ast import Module
from bdb import effective
from cProfile import label
from functools import partial
from black import Mode
from matplotlib.pyplot import grid
from dataclasses import dataclass

import warnings
import math
from copy import deepcopy
from typing import Union, Tuple, Sequence, Optional, List, Callable, Set
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out
from torchvision.utils import save_image

from transformers.activations import PytorchGELUTanh, ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, BaseModelOutputWithPooling, BaseModelOutput
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    can_return_tuple,
    logging,
    replace_return_docstrings,
    torch_int,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    ViT_Attention_type = "flash_attention_2"
else:
    flash_attn_varlen_func = None
    ViT_Attention_type = "sdpa"
from transformers.activations import ACT2FN
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin

from .configuration_umm import Qwen2Config, Siglip2VisionConfig, UMMConfig
logger = logging.get_logger(__name__)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention.flex_attention import BlockMask


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IMAGE_GEN_TOKEN_INDEX = -300

IM_START_ID = 151667
IM_END_ID = 151668
NO_MEAN_ID = 151669
EOS_TOKEN_ID = 151645

@dataclass
class Siglip2VisionOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
    
def eager_attention_forward_siglip2(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[Siglip2VisionConfig]):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        # self.is_causal = True

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward_siglip2
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                attention_interface = ALL_ATTENTION_FUNCTIONS[ViT_Attention_type]
        
        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Union[Siglip2VisionConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2Config
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    @can_return_tuple
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Siglip2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state

        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)

def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)

def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")

def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")

class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]
    
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * config.z_channels, 2 * config.z_channels, 1)
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        # downsampling
        self.conv_in = nn.Conv2d(config.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * config.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(config.z_channels, config.z_channels, 1)
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, config.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)
        if type(attention_mask) == BlockMask:
            attn_output = flex_attention(query_states, key_states, value_states, block_mask=attention_mask)
        else:
            causal_mask = causal_mask.to(torch.bool).to(value_states.device)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "sdpa": Qwen2SdpaAttention,
}

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

@dataclass
class UMMCausalLMOutput(CausalLMOutputWithPast):
    image_latent: Optional[torch.FloatTensor] = None
    image_latent_label: Optional[torch.FloatTensor] = None
    image_pixcel: Optional[torch.FloatTensor] = None
    image_pixcel_label: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    image_loss: Optional[torch.FloatTensor] = None
    time_embeds_1: Optional[torch.FloatTensor] = None
    time_embeds_2: Optional[torch.FloatTensor] = None

class UMMPretrainedModel(PreTrainedModel):
    config: UMMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "Siglip2EncoderLayer", "Encoder", "Decoder"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

class VAEModel(UMMPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config.vae_encoder_config)
        self.decoder = Decoder(config.vae_decoder_config)
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * config.vae_encoder_config.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.post_init()

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)
    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m
    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z
    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec

class VAEDecoderProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config.vae_decoder_config)
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * config.vae_encoder_config.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
    
    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def forward(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec
    
class UMMUndProjector(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_embed_dim,
        compression_factor=(2,2),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embed_dim = image_embed_dim
        self.layernorm = nn.LayerNorm(image_embed_dim)
        self.hidden_size = image_embed_dim * (compression_factor[0]*compression_factor[1])
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, embed_dim),
        )
        self.compression_factor = compression_factor
    
    def forward(self, x):
        x = x.to(torch.bfloat16)
        x = self.layernorm(x)
        height, width = int(x.size(1)**0.5), int(x.size(1)**0.5)
        x = x.permute(0, 2, 1).unflatten(-1, (height, width))  # b, dim, h, w
        batch_size, dim, height, width = x.shape
        unfolded = x.unfold(2, self.compression_factor[0], self.compression_factor[0]).unfold(3, self.compression_factor[1], self.compression_factor[1])
        unfolded = unfolded.contiguous().view(batch_size, dim, -1, self.compression_factor[0] * self.compression_factor[1])
        unfolded = unfolded.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, dim*self.compression_factor[0] * self.compression_factor[1]) 
        compressed_x = self.mlp(unfolded)
        return compressed_x
    
class DiTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            hidden_size, 
            layer_idx,
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_dropout=0.0, 
            attn_implementation='sdpa'
            ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.attn_implementation = attn_implementation
        self.is_causal = False
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation] 
        attention_interface = ALL_ATTENTION_FUNCTIONS[ViT_Attention_type]     
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def modulate(x, shift, scale):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    shift = shift.to(torch.float32)
    scale = scale.to(torch.float32)
    if len(x.shape) != len(shift.shape):
        return (x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)).to(input_dtype)
    else:
        return (x * (1 + scale) + shift).to(input_dtype)

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class ModulatedAttentionBlock(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            layer_idx,
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_dropout=0.0,
            ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = DiTAttention(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
        )

        self.mlp = MLP(self.hidden_size, self.hidden_size*4)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.hidden_size,
                6*self.hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
    
    def forward(
            self, 
            hidden_states, 
            adaln_input, 
            attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(6, dim=-1)
        residual = hidden_states
        hidden_states = modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        hidden_states, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + gate_msa * hidden_states

        residual = hidden_states
        hidden_states = modulate(self.post_attention_layernorm(hidden_states), shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs

class DiTRotaryEmbedding(nn.Module):
    def __init__(
            self, 
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            scaling_factor=1.0,):
        super().__init__()
        self.rope_kwargs={
            "rope_type": "default",
            "factor": scaling_factor,
            "dim": dim,
            "base": base,
            "max_position_embeddings": max_position_embeddings,
        }
        self.rope_type = "default"
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.config = None
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, None, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size*patch_size*out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )
    
    def forward(self, x, adaln_input):
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class UMMGenProjector(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_attention_heads,
        num_key_value_heads,
        patch_size,
        output_dim,
        layers_num,
    ):
        super().__init__()
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(hidden_size=embed_dim, layer_idx=layer_idx, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads) for layer_idx in range(layers_num)]
        )
        self.diffusion_head_b = FinalLayer(hidden_size=embed_dim, patch_size=patch_size, out_channels=output_dim)
        self.rotary_emb = DiTRotaryEmbedding(
            dim=embed_dim // num_attention_heads,
            max_position_embeddings=2048,
            base=10000,
            scaling_factor=1.0,
        )
    
    def forward(self, x, time_embeds, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device). unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        hidden_states = x
        for layer in self.diffusion_head_a:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                adaln_input=time_embeds,
                position_ids=position_ids,
            )[0]
        v_pred = self.diffusion_head_b(hidden_states, time_embeds)
        return v_pred

class UMMGenHiProjector(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_attention_heads,
        num_key_value_heads,
        patch_size,
        output_dim,
        layers_num,
    ):
        super().__init__()
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(hidden_size=embed_dim, layer_idx=layer_idx, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads) for layer_idx in range(layers_num)]
        )
        self.diffusion_head_b = FinalLayer(hidden_size=embed_dim, patch_size=patch_size, out_channels=output_dim)
        self.rotary_emb = DiTRotaryEmbedding(
            dim=embed_dim // num_attention_heads,
            max_position_embeddings=2048,
            base=10000,
            scaling_factor=1.0,
        )
    
    def forward(self, x, time_embeds):
        position_ids = torch.arange(x.shape[1] // 2, device=x.device). unsqueeze(0)
        cos, sin = self.rotary_emb(x, position_ids)
        cos = cos.repeat(1, 2, 1)
        sin = sin.repeat(1, 2, 1)
        position_embeddings = (cos, sin)
        hidden_states = x
        for layer in self.diffusion_head_a:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                adaln_input=time_embeds,
                position_ids=position_ids,
            )[0]
        v_pred = self.diffusion_head_b(hidden_states, time_embeds)
        return v_pred

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, frequency_embedding_size=256):
        super().__init__()
        self.mlp_1 = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size_1, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size_1, hidden_size_1, bias=True),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size_2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size_2, hidden_size_2, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb_1 = self.mlp_1(t_freq)
        t_emb_2 = self.mlp_2(t_freq)
        return t_emb_1, t_emb_2

class HiGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim,1),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
    
    def forward(self, low_info, high_info, heat_map=False):
        hi_gate = self.gate(low_info)
        output = low_info + hi_gate * high_info
        output = self.layer_norm(output)
        if heat_map:
            hi_gate_map = hi_gate[0, :, 0]
            seq_len = hi_gate_map.shape[0]
            H=W=int(math.sqrt(seq_len))
            heat_map = hi_gate_map.view(H, W).detach().float().cpu().numpy()
            return output, heat_map
        return output

class UMMTextModel(UMMPretrainedModel):
    config: Qwen2Config
    _no_split_modules = ["Qwen2DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = attention_mask
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class UMMModel(UMMPretrainedModel):
    config_class = UMMConfig
    _no_split_modules = [
        "Encoder",
        "Decoder",
        "Siglip2EncoderLayer",
        "UMMUndProjector",
        "UMMGenProjector",
        "UMMGenHiProjector",
        "Qwen2DecoderLayer"
    ]
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_representation = Siglip2VisionTransformer(config.vision_representation_config)
        self.vae_model = VAEModel(config)
        self.vae_decoder_projector = VAEDecoderProjector(config)
        text_embed_dim = config.text_config.hidden_size
        image_embed_dim = config.vision_representation_config.hidden_size
        t_num_attention_heads = config.text_config.num_attention_heads
        t_num_key_value_heads = config.text_config.num_key_value_heads
        i_num_attention_heads = config.vision_representation_config.num_attention_heads
        i_num_key_value_heads = config.text_config.num_key_value_heads   
        self.time_embed = TimestepEmbedder(hidden_size_1=text_embed_dim, hidden_size_2=image_embed_dim)
        self.und_projector = UMMUndProjector(embed_dim=text_embed_dim, image_embed_dim=image_embed_dim)
        self.gen_projector = UMMGenProjector(
                                            embed_dim=text_embed_dim, 
                                            num_attention_heads=t_num_attention_heads, 
                                            num_key_value_heads=t_num_key_value_heads,
                                            patch_size=2,
                                            output_dim=image_embed_dim,
                                            layers_num=7,
                                            )
        self.hi_gate = HiGate(embed_dim=image_embed_dim)
        self.hi_projector = UMMGenHiProjector(
                                            embed_dim=image_embed_dim, 
                                            num_attention_heads=i_num_attention_heads, 
                                            num_key_value_heads=i_num_key_value_heads,
                                            patch_size=1,
                                            output_dim=config.vae_decoder_config.ch,
                                            layers_num=3,
                                            )
        self.language_model = UMMTextModel._from_config(config.text_config)
        self.post_init()
    
    def path_sample(self, t, x0, x1):
        dims = [1] * len(x1[0].size())
        t = t.view(t.size(0), *dims)
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1-t, -1
        xt = alpha_t * x1 + sigma_t * x0
        ut = d_alpha_t * x1 + d_sigma_t * x0
        mask = (t < 1).float().to(device=ut.device, dtype=ut.dtype)
        ut = ut * mask
        return xt, ut
    
    def get_image_features(self, pixel_values, t=None):
        pixel_values = pixel_values.to(self.vae_model.encoder.quant_conv.weight.dtype)
        with torch.no_grad():
            image_latent = self.vae_model.encode(pixel_values)  ### b, 128, h/16, w/16

        x0 = torch.randn_like(image_latent)
        if t is not None:
            try:
                xt, ut = self.path_sample(t, x0, image_latent)
            except:
                xt, ut = image_latent, torch.zeros_like(image_latent, device=image_latent.device, dtype=image_latent.dtype)
        else:
            xt, ut = image_latent, torch.zeros_like(image_latent, device=image_latent.device, dtype=image_latent.dtype)

        xt = xt.to(torch.bfloat16)
        image_pixel_hat = self.vae_decoder_projector(xt)
        if image_pixel_hat.size(-1) > 512:
            interpolate_pos_encoding = True
        else:
            interpolate_pos_encoding = False
        image_features = self.vision_representation(image_pixel_hat, interpolate_pos_encoding=interpolate_pos_encoding).last_hidden_state
        image_target = ut.reshape(ut.size(0), ut.size(1), -1).permute(0, 2, 1)
        projected_image_features = self.und_projector(image_features)
        output = {
            "image_target": image_target,
            "projected_image_features": projected_image_features,
            "pixel_values": pixel_values,
            "image_pixel_hat": image_pixel_hat,
        }
        return output

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        t,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        pixel_values=None,
        grid_hws=None
    ):
        if pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, None, None, None, None
        image_features = self.get_image_features(pixel_values, t)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        image_gen_pixcel_hat = []
        new_input_embeds = []
        new_labels = []
        image_gen_labels = []
        image_gen_pixcel_labels = []
        image_mask = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_images_gen = (cur_input_ids == IMAGE_GEN_TOKEN_INDEX).sum()
            if num_images == 0 and num_images_gen == 0:
                cur_new_labels = []
                cur_image_features = image_features["projected_image_features"][cur_image_idx]
                cur_input_embeds_1 = self.language_model.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_image_features, cur_input_embeds_1], dim=0)
                cur_image_targets = torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=labels[batch_idx].device, dtype=labels[batch_idx].dtype)
                new_input_embeds.append(cur_input_embeds)
                image_gen_labels.append(image_features["image_target"][cur_image_idx])
                image_gen_pixcel_labels.append(image_features["pixel_values"][cur_image_idx])
                image_gen_pixcel_hat.append(image_features["image_pixel_hat"][cur_image_idx])
                image_mask.append(torch.zeros(cur_input_embeds.size(0), dtype=torch.bool))
                cur_new_labels.append(cur_image_targets)
                cur_new_labels.append(labels[batch_idx])
                cur_new_labels = torch.cat(cur_new_labels)
                new_labels.append(cur_new_labels)
                cur_image_idx += 1
                continue
            
            image_token_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_gen_token_positions = torch.where(cur_input_ids == IMAGE_GEN_TOKEN_INDEX)[0].tolist()

            all_insert_positions = sorted([
                (pos, "image") for pos in image_token_positions
            ] + [
                (pos, "image_gen") for pos in image_gen_token_positions
            ], key=lambda x: x[0])

            all_token_indices = [-1] + [p[0] for p in all_insert_positions] + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(all_token_indices) - 1):
                start = all_token_indices[i] + 1
                end = all_token_indices[i + 1]
                cur_input_ids_noim.append(cur_input_ids[start: end])
                cur_labels_noim.append(cur_labels[start: end])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []   
            cur_new_labels = []
            cur_image_mask = []

            for i in range(len(cur_input_embeds_no_im)):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_image_mask.append(torch.zeros(cur_input_embeds_no_im[i].size(0), dtype=torch.bool))

                if i < len(all_insert_positions):
                    token_type = all_insert_positions[i][1]
                    if token_type == "image":
                        cur_image_features = image_features["projected_image_features"][cur_image_idx]
                        cur_image_targets = torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                        image_gen_labels.append(image_features["image_target"][cur_image_idx])
                        image_gen_pixcel_labels.append(image_features["pixel_values"][cur_image_idx])
                        image_gen_pixcel_hat.append(image_features["image_pixel_hat"][cur_image_idx])
                        cur_image_idx += 1
                    elif token_type == "image_gen":
                        cur_image_features = image_features["projected_image_features"][cur_image_idx]
                        cur_image_targets = torch.full((cur_image_features.shape[0],), IMAGE_GEN_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                        image_gen_labels.append(image_features["image_target"][cur_image_idx])
                        image_gen_pixcel_labels.append(image_features["pixel_values"][cur_image_idx])
                        image_gen_pixcel_hat.append(image_features["image_pixel_hat"][cur_image_idx])
                        cur_image_idx += 1
                    else:
                        raise ValueError(f"Unexpected token type: {token_type}")
                    
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(cur_image_targets)
                    cur_image_mask.append(torch.ones(cur_image_features.size(0), dtype=torch.bool))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_image_mask = torch.cat(cur_image_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            image_mask.append(cur_image_mask)


        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", 4096)
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        image_mask = [x[:tokenizer_model_max_length] for x in image_mask]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_image_mask = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_image_mask) in enumerate(zip(new_input_embeds, new_labels, image_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_image_mask[i, -cur_len:] = cur_image_mask
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_image_mask[i, :cur_len] = cur_image_mask

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_gen_labels, image_gen_pixcel_labels, image_gen_pixcel_hat, new_image_mask

    def forward(
        self, 
        input_ids = None, 
        t = None,
        position_ids = None,
        attention_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        pixel_values = None,
        grid_hws = None,
        return_dict = None,
        **kwargs,
        ):

        if inputs_embeds is None or pixel_values is not None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_gen_labels, image_gen_pixcel_labels, image_gen_pixcel_hat, new_image_mask = self.prepare_inputs_labels_for_multimodal(
                input_ids, t, position_ids, attention_mask, past_key_values, labels, pixel_values, grid_hws
            )
        device = inputs_embeds.device
        if t is not None and len(t) == len(pixel_values):
            t_embeds_1, t_embeds_2 = self.time_embed(t, inputs_embeds.dtype)
        elif isinstance(t, list):
            t = torch.cat(t).unsqueeze(1).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            t_embeds_1, t_embeds_2 = self.time_embed(t, inputs_embeds.dtype)
        else:
            t = torch.ones((len(pixel_values), 1), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            t_embeds_1, t_embeds_2 = self.time_embed(t, inputs_embeds.dtype)

        batch_size, seq_len = new_image_mask.shape
        head_num = self.config.text_config.num_attention_heads
        omni_attention_mask = torch.tril(torch.ones((batch_size,head_num,seq_len,seq_len), dtype=torch.long)).to(device)

        img = new_image_mask.bool()
        seg = (img & ~torch.nn.functional.pad(img[:, :-1], (1, 0), value=False)).cumsum(1) * img
        attention_mask_img = ((seg.unsqueeze(2) == seg.unsqueeze(1)) & img.unsqueeze(2) & img.unsqueeze(1))
        attention_mask_img = attention_mask_img.to(dtype=torch.long).to(device)

        txt = attention_mask.bool()
        seg = (txt & ~torch.nn.functional.pad(txt[:, :-1], (1, 0), value=False)).cumsum(1) * txt
        attention_mask_txt = ((seg.unsqueeze(2) == seg.unsqueeze(1)) & txt.unsqueeze(2) & txt.unsqueeze(1))
        attention_mask_txt = attention_mask_txt.to(dtype=torch.long).to(device)

        pad = ~attention_mask.bool()
        seg = (pad & ~torch.nn.functional.pad(pad[:, :-1], (1, 0), value=False)).cumsum(1) * pad
        attention_mask_pad = ((seg.unsqueeze(2) == seg.unsqueeze(1)) & pad.unsqueeze(2) & pad.unsqueeze(1))
        attention_mask_pad = attention_mask_pad.to(dtype=torch.long).to(device)

        for i in range(omni_attention_mask.size(1)):
            omni_attention_mask[:,i,:,:] = torch.bitwise_or(omni_attention_mask[:,i,:,:], attention_mask_img) * attention_mask_txt
            omni_attention_mask[:,i,:,:] = torch.bitwise_or(omni_attention_mask[:,i,:,:], attention_mask_pad)
        output = self.language_model(
            input_ids=None,
            attention_mask=omni_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if labels is not None:
            return output, t_embeds_1, t_embeds_2, labels, image_gen_labels, image_gen_pixcel_labels, image_gen_pixcel_hat
        return output, t_embeds_1, t_embeds_2, new_image_mask

class Cheers(UMMPretrainedModel, GenerationMixin):
    config_class = UMMConfig
    _no_split_modules = [
        "Encoder",
        "Decoder",
        "Siglip2EncoderLayer",
        "UMMUndProjector",
        "UMMGenProjector",
        "Qwen2DecoderLayer"
    ]

    _tied_weights_keys = ["lm_head.weight", "model.language_model.embed_tokens.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config.text_config._attn_implementation = self.config._attn_implementation
        self.config.vision_representation_config._attn_implementation = self.config._attn_implementation
        
        self.model = UMMModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.text_loss_fc = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.post_init()
    
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vae_model(self):
        return self.model.vae_model
    
    @property
    def vision_representation(self):
        return self.model.vision_representation

    def get_input_embeddings(self):
        return self.language_model.embed_tokens
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def forward(self, input_ids, t=None, labels=None, attention_mask=None, pixel_values=None, grid_hws=None, **kwargs):
        if grid_hws is not None:
            image_h, image_w = grid_hws[0]
            per_image_token = int(image_h * image_w) // 4
        else:
            image_h, image_w = 32, 32
            per_image_token = 256
        if labels is not None:
            outputs, t_embeds_1, t_embeds_2, labels, image_gen_labels, image_gen_pixcel_labels, image_gen_pixcel_hat = self.model(input_ids, t=t, labels=labels, attention_mask=attention_mask, pixel_values=pixel_values, grid_hws=grid_hws, **kwargs)
            hidden_states = outputs[0][:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            bsz, seq_len, dim = hidden_states.shape
            device = hidden_states.device
            image_num = len(image_gen_labels)
            mask_img_de = torch.zeros_like(labels, dtype=torch.bool)
            image_latent = None
            for b in range(bsz):
                cur_labels = labels[b]
                new_cur_labels = []
                for idx, cur_id in enumerate(cur_labels):
                    if cur_id == IM_START_ID:
                        new_cur_labels.append(IM_START_ID)
                        new_cur_labels.append(-100)
                    elif cur_id == IM_END_ID:
                        continue 
                    elif cur_id == IMAGE_TOKEN_INDEX or cur_id == IMAGE_GEN_TOKEN_INDEX:
                        new_cur_labels.append(-100)
                    else:
                        new_cur_labels.append(cur_id)
                new_cur_labels = torch.tensor(new_cur_labels, dtype=cur_labels.dtype, device=cur_labels.device)
                labels[b] = new_cur_labels

                start_indices = (new_cur_labels == IM_START_ID).nonzero(as_tuple=True)[0]
                for start in start_indices:
                    mask_img_de[b, start + 2:start + per_image_token + 2] = True
                if not mask_img_de[b].any():
                    mask_img_de[b, :per_image_token] = True
            mask_text = (labels != -100)
            mask_text[mask_img_de] = False
            anchor_text = (self.lm_head(hidden_states[:, :1, :]).sum()) * 0.0
            anchor_img = hidden_states[:, :1, :].sum() * 0.0
            zero_anchor = anchor_text + anchor_img

            if mask_text.any():
                out_text = self.lm_head(hidden_states)
            if mask_img_de.any():
                h_img_de = hidden_states[mask_img_de]

                try:
                    h_img_de = h_img_de.reshape(image_num, per_image_token, h_img_de.size(-1))
                except:
                    print(h_img_de.shape, image_num, per_image_token, mask_img_de.shape)
                image_latent = self.model.gen_projector(h_img_de, t_embeds_1)
                image_latent = image_latent.reshape(image_num, int(image_h//2), int(image_w//2), image_latent.size(-1))

                B, H, W, C = image_latent.shape
                P = 2
                D = C//(P*P)
                image_latent = image_latent.view(B, H, W, P, P, D)
                image_latent = image_latent.permute(0, 1, 3, 2, 4, 5).contiguous()
                image_latent = image_latent.view(B, H*P, W*P, D)
                image_latent = image_latent.view(B, H*P*W*P, D)

                image_gen_pixcel_hat = torch.stack(image_gen_pixcel_hat, dim=0)
                patch_embedding_res = self.model.vision_representation.embeddings(image_gen_pixcel_hat)

                hi_input = self.model.hi_gate(image_latent, patch_embedding_res)
                image_latent_pre = self.model.hi_projector(hi_input, t_embeds_2)
                image_latent = image_latent_pre.view(B, image_h, image_w, image_latent_pre.size(-1))
                image_latent = image_latent.permute(0, 3, 1, 2)

            text_loss_denominator = (labels != -100).sum().clamp(min=1)
            image_loss_denominator = mask_img_de.sum()

            if text_loss_denominator.item() > 0:
                text_logits = out_text.view(-1, out_text.shape[-1]).contiguous()
                text_labels = labels.view(-1).type(torch.long).contiguous()
                text_loss = self.text_loss_fc(text_logits, text_labels)
                valid_mask = (text_labels != -100).type_as(text_loss)
                text_loss = text_loss * valid_mask
                text_loss = text_loss.sum() / text_loss_denominator
            else:
                text_loss = zero_anchor
            
            if image_loss_denominator.item() > 0:
                #### latent space loss ####
                image_gen_labels = torch.stack(image_gen_labels, dim=0)
                image_loss = F.mse_loss(image_latent_pre, image_gen_labels)
            else:
                image_loss = zero_anchor
            
            alpha = 1.0
            total_loss = text_loss + alpha * image_loss
            image_latent_label = image_gen_labels
            if image_latent is not None and image_latent.numel():
                with torch.no_grad():
                    image_pixcel = self.model.vae_model.decode(image_latent)
            else:
                image_pixcel = None

            return UMMCausalLMOutput(
                loss=total_loss,
                logits=text_logits,
                image_latent=image_latent,
                image_latent_label=image_latent_label,
                image_pixcel=image_pixcel,
                image_pixcel_label=image_gen_pixcel_labels,
                text_loss=text_loss,
                image_loss=image_loss,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.last_hidden_state,
                attentions=outputs.attentions
            )

        else:
            outputs, t_embeds_1, t_embeds_2, omni_attention_mask = self.model(input_ids, t=t, labels=labels, attention_mask=attention_mask, pixel_values=pixel_values, grid_hws=grid_hws, **kwargs)
            hidden_states = outputs.last_hidden_state
            slice_indices = slice(0, None)
            logits = self.lm_head(hidden_states[:,slice_indices,:])
            loss = None
        
            return UMMCausalLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=omni_attention_mask,
                time_embeds_1=t_embeds_1,
                time_embeds_2=t_embeds_2,
            )
    
    def generate(
        self, 
        input_ids, 
        max_length=2048,
        temperature=0.7,
        top_p=1.0,
        t=None, 
        cfg_scale=None, 
        attention_mask=None, 
        pixel_values=None, 
        grid_hws=None, 
        num_inference_steps=50,
        num_beams=1,
        length_penalty=1.0,
        repetition_penalty=1.1,
        edit_image=False,
        alpha=1.0,
        **kwargs
    ):
        if temperature != 0 and num_beams > 1:
            warnings.warn(
                "Both temperature != 0 and num_beams > 1 are set. "
                "Beam search will be used and sampling paramters are ignored.",
                UserWarning
            )
        max_input_tokens = 1024
        max_output_tokens = max_length
        orig_len = input_ids.size(1)
        if grid_hws is not None:
            image_h, image_w = grid_hws[0]
            per_image_token = int(image_h * image_w) // 4
        else:
            image_h, image_w = 32, 32
            per_image_token = 1024 // 4

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        ones = torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)  

        outputs = self(
            input_ids, 
            t=t, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            grid_hws=grid_hws,
            use_cache=True,
            output_hidden_states=True, 
        )
        past_key_values = outputs.past_key_values
        last_input_ids = input_ids[0][-1]
        last_token_hidden_states = outputs.hidden_states[:,-1,:].unsqueeze(1)

        if cfg_scale is not None and cfg_scale != 1.0:
            use_cfg = True
            uncond_input_ids = self._build_uncond_full_ids(input_ids)
            uncond_outputs = self(
                uncond_input_ids, 
                t=t, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                grid_hws=grid_hws,
                use_cache=True,
                output_hidden_states=True,
            )
            uncond_last_input_ids = uncond_input_ids[0][-1]
            uncond_past_key_values = uncond_outputs.past_key_values
            uncond_last_token_hidden_states = uncond_outputs.hidden_states[:,-1,:].unsqueeze(1)
        else:
            use_cfg = False

        def _clone_dynamic_cache(cache):
            legacy = cache.to_legacy_cache()
            legacy = tuple(tuple(x.clone() for x in layer) for layer in legacy)
            return DynamicCache.from_legacy_cache(legacy)
        
        image_latent_pre_list = []
        image_gen_time = False
        cur_time = 0
        attention_mask = outputs.attentions
        while True:
            if input_ids.size(1) > max_input_tokens:
                break
            if cur_time > max_output_tokens:
                break
            if last_input_ids == IM_START_ID:   
                image_gen_time = True
            
            if image_gen_time:
                if pixel_values is not None and len(pixel_values) == input_ids.size(0) and edit_image == True:
                    start_t = 0.1
                    num_inference_steps = int(num_inference_steps * (1-start_t))
                    last_step_size = 1 / num_inference_steps 
                    t_list = torch.linspace(start_t, 1, num_inference_steps)
                    t_list = self.time_shift(t_list, alpha=alpha)
                    z_image = self.model.vae_model.encode(pixel_values.to(dtype=last_token_hidden_states.dtype, device=last_token_hidden_states.device))
                    noise_z = torch.randn((input_ids.size(0), 128, image_h, image_w), dtype=last_token_hidden_states.dtype, device=last_token_hidden_states.device)
                    z = start_t * z_image + (1 - start_t) * noise_z
                else:
                    last_step_size = 1 / num_inference_steps 
                    t_list = torch.linspace(0, 1, num_inference_steps)
                    t_list = self.time_shift(t_list, alpha=alpha)
                    z = torch.randn((input_ids.size(0), 128, image_h, image_w), dtype=last_token_hidden_states.dtype, device=last_token_hidden_states.device) 
                z_mask = torch.ones((z.size(0), (image_h//2) *(image_w//2)), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, z_mask], dim=1)
                x_t = z
                for n in range(num_inference_steps): 
                    ti = t_list[n]
                    with torch.no_grad():
                        drift, step_output, heat_map = self._drift_fn(x_t, ti, attention_mask, past_key_values, out_heat_map=True)
                        if use_cfg:
                            uncond_drift, uncond_step_output = self._drift_fn(x_t, ti, attention_mask, uncond_past_key_values)
                            drift = uncond_drift + cfg_scale * (drift - uncond_drift)
                        if ti != 1 and n < num_inference_steps-1:
                            dt = t_list[n+1] - t_list[n]
                            x_t = self.euler_maruyama_step(x_t, drift, dt)
                            past_key_values.crop(-per_image_token)   
                            if use_cfg:
                                uncond_past_key_values.crop(-per_image_token)
                        else:
                            image_latent_pre = x_t + drift * last_step_size

                past_key_values = step_output.past_key_values
                if use_cfg:
                    uncond_past_key_values = uncond_step_output.past_key_values
                image_latent_pre_list.append(image_latent_pre)

                attention_mask = torch.cat([attention_mask, ones], dim=1)
                image_gen_time = False
                last_input_ids = IM_END_ID
                last_input_ids_tensor = torch.full((input_ids.size(0),1), last_input_ids, dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, last_input_ids_tensor], dim=1)
                inputs_embeds = self.language_model.embed_tokens(last_input_ids_tensor)
                text_attention_mask = torch.ones((1, 1, 1, attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
                step_output = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=text_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = step_output.past_key_values
                last_token_hidden_states = step_output.last_hidden_state[:,-1,:].unsqueeze(1)
                if use_cfg:
                    uncond_last_input_ids = IM_END_ID
                    uncond_last_input_ids_tensor = torch.full((input_ids.size(0),1), uncond_last_input_ids, dtype=input_ids.dtype, device=input_ids.device)
                    uncond_input_ids = torch.cat([uncond_input_ids, uncond_last_input_ids_tensor], dim=1)
                    uncond_inputs_embeds = self.language_model.embed_tokens(uncond_last_input_ids_tensor)
                    text_attention_mask = torch.ones((1, 1, 1, attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
                    step_output = self.language_model(
                        inputs_embeds=uncond_inputs_embeds,
                        attention_mask=text_attention_mask,
                        past_key_values=uncond_past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    uncond_past_key_values = step_output.past_key_values
                    uncond_last_token_hidden_states = step_output.last_hidden_state[:,-1,:].unsqueeze(1)
                cur_time += per_image_token
            else:

                if last_input_ids == EOS_TOKEN_ID:
                    break
                if num_beams is None or num_beams <=1:

                    last_logits = self.lm_head(last_token_hidden_states)
                    last_input_ids = self._sample_from_logits(last_logits[:,-1,:], temp=temperature, top_p=top_p)
                    last_input_ids_tensor = last_input_ids.unsqueeze(1)
                    last_input_ids = last_input_ids[0]
                    input_ids = torch.cat([input_ids, last_input_ids_tensor], dim=1)
                    if use_cfg:
                        uncond_last_logits = self.lm_head(uncond_last_token_hidden_states)
                        uncond_last_input_ids = self._sample_from_logits(uncond_last_logits[:,-1,:], temp=temperature, top_p=top_p)
                        uncond_last_input_ids_tensor = uncond_last_input_ids.unsqueeze(1)
                        uncond_last_input_ids = uncond_last_input_ids[0]
                        uncond_input_ids = torch.cat([uncond_input_ids, uncond_last_input_ids_tensor], dim=1)

                    attention_mask = torch.cat([attention_mask, ones], dim=1)
                    inputs_embeds = self.language_model.embed_tokens(last_input_ids_tensor)
                    text_attention_mask = torch.ones((1, 1, 1, attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
                    step_output = self.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=text_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_key_values = step_output.past_key_values
                    last_token_hidden_states = step_output.last_hidden_state[:,-1,:].unsqueeze(1)
                    if use_cfg:
                        uncond_inputs_embeds = self.language_model.embed_tokens(uncond_last_input_ids_tensor)
                        text_attention_mask = torch.ones((1, 1, 1, attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
                        step_output = self.language_model(
                            inputs_embeds=uncond_inputs_embeds,
                            attention_mask=text_attention_mask,
                            past_key_values=uncond_past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        uncond_past_key_values = step_output.past_key_values
                        uncond_last_token_hidden_states = step_output.last_hidden_state[:,-1,:].unsqueeze(1)
                else:
                    assert input_ids.size(0) == 1, "only support beam=1"
                    if 'beam_state' not in locals():
                        beam_state = {
                            "seqs":[input_ids.clone() for _ in range(num_beams)],
                            "scores": torch.zeros(num_beams, device=input_ids.device),
                            "pkv": [_clone_dynamic_cache(past_key_values) for _ in range(num_beams)],
                            "h": [last_token_hidden_states.clone() for _ in range(num_beams)],
                            "finished": torch.zeros(num_beams, dtype=torch.bool, device=input_ids.device),
                        }
                    k_expand = num_beams
                    
                    all_cand = []  # (new_score, beam_idx, token_id)
                    for bi in range(num_beams):
                        if beam_state["finished"][bi]:
                            all_cand.append((beam_state["scores"][bi], bi, EOS_TOKEN_ID))
                            continue

                        h = beam_state["h"][bi]                      # (1,1,D)
                        logits = self.lm_head(h)[:, -1, :].float()   # (1,V)
                        logprobs = torch.log_softmax(logits, dim=-1) # (1,V)
                        rep = repetition_penalty
                        if rep is not None and rep != 1.0:
                            prev = beam_state["seqs"][bi][0]
                            prev_ids = prev.unique()
                            lp_vals = logprobs[0, prev_ids]
                            logprobs[0, prev_ids] = torch.where(lp_vals > 0, lp_vals / rep, lp_vals * rep)
                        topk_lp, topk_id = torch.topk(logprobs, k_expand, dim=-1)
                        for j in range(k_expand):
                            tok = int(topk_id[0, j].item())
                            sc = beam_state["scores"][bi] + topk_lp[0, j]
                            all_cand.append((sc, bi, tok))

                    all_cand.sort(key=lambda x: x[0].item(), reverse=True)
                    new_cand = all_cand[:num_beams]

                    new_seqs, new_pkv, new_h = [], [], []
                    new_scores = torch.empty(num_beams, device=input_ids.device)
                    new_finished = torch.empty(num_beams, dtype=torch.bool, device=input_ids.device)

                    for i, (sc, parent_bi, tok) in enumerate(new_cand):
                        new_scores[i] = sc

                        parent_seq = beam_state["seqs"][parent_bi]
                        tok_tensor = torch.tensor([[tok]], dtype=parent_seq.dtype, device=parent_seq.device)
                        seq_i = torch.cat([parent_seq, tok_tensor], dim=1)
                        new_seqs.append(seq_i)

                        if tok == EOS_TOKEN_ID:
                            new_pkv.append(beam_state["pkv"][parent_bi])
                            new_h.append(beam_state["h"][parent_bi])
                            new_finished[i] = True
                            continue

                        inputs_embeds = self.language_model.embed_tokens(tok_tensor)
                        past_len = beam_state["pkv"][parent_bi].get_seq_length()
                        cache_position = torch.tensor([past_len], device=inputs_embeds.device, dtype=torch.long)
                        text_attention_mask = torch.ones((1, 1, 1, past_len+1), dtype=attention_mask.dtype, device=attention_mask.device)

                        step_output = self.language_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=text_attention_mask,
                            past_key_values=beam_state["pkv"][parent_bi],
                            cache_position=cache_position,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        new_pkv.append(step_output.past_key_values)
                        new_h.append(step_output.last_hidden_state[:, -1, :].unsqueeze(1))
                        new_finished[i] = False


                    lengths = torch.tensor([s.size(1) for s in new_seqs], device=input_ids.device, dtype=torch.float32)
                    if length_penalty is not None and length_penalty != 1.0:
                        norm_scores = new_scores / (lengths ** length_penalty)
                    else:
                        norm_scores = new_scores
                    best_idx = int(torch.argmax(norm_scores).item())

                    beam_state["seqs"] = new_seqs
                    beam_state["pkv"] = new_pkv
                    beam_state["h"] = new_h
                    beam_state["scores"] = new_scores
                    beam_state["finished"] = new_finished


                    input_ids = beam_state["seqs"][best_idx]
                    past_key_values = beam_state["pkv"][best_idx]
                    last_token_hidden_states = beam_state["h"][best_idx]
                    last_input_ids = int(input_ids[0, -1].item())

                    attention_mask = torch.cat([attention_mask, ones], dim=1)
                cur_time += 1

        all_images = []
        for image_latent in image_latent_pre_list:
            image_pixcel = self.model.vae_model.decode(image_latent)
            all_images.append(image_pixcel)
        generated_ids = input_ids[:, orig_len:]
        return {
            "input_ids": generated_ids,
            "images": all_images,
        }

    def time_shift(self, ts, alpha=1.0):
        return (alpha * ts) / (1.0 + (alpha-1.0) * ts)

    def _drift_fn(self, x_t, t, attention_mask, past_key_values, out_heat_map=False):
        t = torch.full((x_t.size(0),1), t, device=x_t.device, dtype=x_t.dtype)
        t_embeds_1, t_embeds_2 = self.model.time_embed(t, t.dtype)
        x_t = self.model.vae_decoder_projector(x_t)
        if x_t.size(-1) > 512:
            interpolate_pos_encoding = True
        else:
            interpolate_pos_encoding = False
        image_feature = self.model.vision_representation(x_t, interpolate_pos_encoding=interpolate_pos_encoding).last_hidden_state
        projected_image_feature = self.model.und_projector(image_feature)
        h_w = int(projected_image_feature.size(1) ** 0.5)

        attention_mask = torch.ones((1, 1, projected_image_feature.size(1), attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
        step_output = self.model.language_model(
            inputs_embeds=projected_image_feature,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        image_feature_pre = step_output.last_hidden_state[:, -projected_image_feature.size(1):, :]
        drift = self.model.gen_projector(image_feature_pre, t_embeds_1)  ### batch_size, 256, dim ###
        drift = drift.reshape(drift.size(0), h_w, h_w, drift.size(-1))

        B, H, W, C = drift.shape
        P = 2
        D = C//(P*P)
        drift = drift.view(B, H, W, P, P, D)
        drift = drift.permute(0, 1, 3, 2, 4, 5).contiguous()
        drift = drift.view(B, H*P, W*P, D)
        drift = drift.view(B, H*P*W*P, D)

        patch_embedding_res = self.model.vision_representation.embeddings(x_t, interpolate_pos_encoding=interpolate_pos_encoding)
        if out_heat_map:
            hi_input, heat_map = self.model.hi_gate(drift, patch_embedding_res, heat_map=out_heat_map)
        else:
            hi_input = self.model.hi_gate(drift, patch_embedding_res, heat_map=out_heat_map)
        drift = self.model.hi_projector(hi_input, t_embeds_2)
        drift = drift.view(B, h_w*2, h_w*2, drift.size(-1))
        drift = drift.permute(0, 3, 1, 2)
        if out_heat_map:
            return drift, step_output, heat_map
        return drift, step_output

    def euler_maruyama_step(self, x, drift, dt):
        mean_x = x + drift * dt
        x = mean_x
        return x
    
    def _build_uncond_full_ids(self, seq_ids):
        bsz, seqlen = seq_ids.shape
        new_ids = seq_ids.clone()
        for b in range(bsz):
            seq = seq_ids[b]
            in_img_block = False
            for t in range(seqlen):
                tok = seq[t].item()
                if tok == IM_START_ID:
                    in_img_block = True
                    new_ids[b, t] = IM_START_ID
                elif tok == IM_END_ID and in_img_block:
                    new_ids[b, t] = IM_END_ID
                    in_img_block = False
                elif tok == IMAGE_TOKEN_INDEX:
                    new_ids[b, t] = IMAGE_TOKEN_INDEX   
                else:
                    if in_img_block:
                        new_ids[b, t] = seq[t]
                    else:
                        new_ids[b, t] = NO_MEAN_ID
        return new_ids

    def _sample_from_logits(
        self,
        logits: torch.Tensor,                 # (B, V)
        temp: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        prev_tokens: torch.Tensor | None = None,  # (B, T)
    ):
        if temp is None or temp <= 1e-6:
            return logits.argmax(dim=-1)

        logits = logits.float()

        if repetition_penalty is not None and repetition_penalty != 1.0 and prev_tokens is not None:
            ids = prev_tokens.long()  # (B, T)
            vals = logits.gather(1, ids)  # (B, T)
            new_vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            logits = logits.scatter(1, ids, new_vals)  

        logits = logits / temp

        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)  # (B,1)
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False  

            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            probs = F.softmax(sorted_logits, dim=-1)

            idx_in_sorted = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            return sorted_idx.gather(1, idx_in_sorted.unsqueeze(-1)).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    
ModelClass = Cheers
__all__ = ["Cheers", "UMMModel", "VAEModel", "UMMPretrainedModel", "UMMTextModel"]