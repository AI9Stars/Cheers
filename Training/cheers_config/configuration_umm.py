from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

class VAEEncoderConfig(PretrainedConfig):
    model_type = "umm"
    base_config_key = "vae_encoder_config"

    def __init__(
        self,
        resolution=256,
        in_channels=3,
        ch=128,
        ch_mult=[1,2,4,4],
        num_res_blocks=2,
        z_channels=32,
        **kwargs
    ):
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        super().__init__(**kwargs)

class VAEDecoderConfig(PretrainedConfig):
    model_type = "umm"
    base_config_key = "vae_decoder_config"

    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1,2,4,4],
        num_res_blocks=2,
        in_channels=3,
        resolution=256,
        z_channels=32,
        **kwargs
    ):
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        super().__init__(**kwargs)

class Siglip2VisionConfig(PretrainedConfig):
    model_type = "umm"
    base_config_key = "vision_representation_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_size = image_size

class Qwen2Config(PretrainedConfig):
    model_type = "umm"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=131072,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class UMMConfig(PretrainedConfig):
    model_type = "umm"
    sub_configs = {
        "vision_representation_config": Siglip2VisionConfig,
        "vae_encoder_config": VAEEncoderConfig,
        "vae_decoder_config": VAEDecoderConfig,
        "text_config": Qwen2Config,
        }
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vision_representation_config=None,
        vae_encoder_config=None,
        vae_decoder_config=None,
        text_config=None,
        **kwargs,
    ):
        if isinstance(vision_representation_config, dict):
            self.vision_representation_config = self.sub_configs["vision_representation_config"](**vision_representation_config)
        elif vision_representation_config is None:
            self.vision_representation_config = self.sub_configs["vision_representation_config"]()

        if isinstance(vae_encoder_config, dict):
            self.vae_encoder_config = self.sub_configs["vae_encoder_config"](**vae_encoder_config)
        elif vae_encoder_config is None:
            self.vae_encoder_config = self.sub_configs["vae_encoder_config"]()
        
        if isinstance(vae_decoder_config, dict):
            self.vae_decoder_config = self.sub_configs["vae_decoder_config"](**vae_decoder_config)
        elif vae_decoder_config is None:
            self.vae_decoder_config = self.sub_configs["vae_decoder_config"]()
        
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__init__(**kwargs)

__all__ = ["UMMConfig"]