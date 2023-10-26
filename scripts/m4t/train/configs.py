# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import yaml

from dataclasses import dataclass
from typing import Dict, Any, Union, get_origin, get_args, List, Literal, Optional


@dataclass
class Config:
    def serialize(self):
        asdict = {}
        for key in self.__dataclass_fields__.keys():
            value = getattr(self, key)
            if isinstance(value, Config):
                asdict[key] = value.serialize()
            else:
                asdict[key] = value
        return asdict

    @classmethod
    def _is_config(cls, type_like: Any) -> bool:
        """Checks if type_like class is a subclass of Config"""
        try:
            if issubclass(type_like, Config):
                return True
        except TypeError:
            pass
        return False

    @classmethod
    def _is_optional_config(cls, type_like: Any) -> bool:
        """Checks if type_like == Optional[subclass of Config]"""
        if not get_origin(type_like) == Union:
            return False
        args = [arg for arg in get_args(type_like) if arg is not type(None)]
        return len(args) == 1 and cls._is_config(args[0])

    @classmethod
    def deserialize(cls, asdict: Dict[str, Any]):
        kwargs = {}
        for key, field_desc in cls.__dataclass_fields__.items():
            non_null = asdict.get(key) is not None
            # Optional[Config]
            if cls._is_optional_config(field_desc.type):
                if non_null:
                    type_arg = [
                        arg
                        for arg in get_args(field_desc.type)
                        if arg is not type(None)
                    ][0]
                    kwargs[key] = type_arg.deserialize(asdict[key])
                else:
                    kwargs[key] = None
            # TODO: add containers with Config
            elif get_origin(field_desc.type) in [Union, List, Dict, Literal]:
                kwargs[key] = asdict.get(key)
            elif cls._is_config(field_desc.type):
                if non_null:
                    kwargs[key] = field_desc.type.deserialize(asdict[key])
                else:
                    kwargs[key] = field_desc.type.default  # type: ignore
            else:
                kwargs[key] = asdict.get(key)
        return cls(**kwargs)

    @classmethod
    def from_string(cls, serialized_config: str):
        return cls.deserialize(yaml.load(serialized_config, Loader=yaml.FullLoader))

    @classmethod
    def from_file(cls, config_path: str):
        return cls.deserialize(yaml.load(config_path, Loader=yaml.FullLoader))


@dataclass
class TextTokenizationConfig(Config):
    from_model: Optional[str] = "seamlessM4T_large"
    """If set, using a tokenizer from the model cards."""

    spm_path: Optional[str] = None
    """Path to a custom spm model. Not used if `from_model` is set."""

    langtoks: Optional[List[str]] = None
    """List of language tokens that should be added. Not used if `from_model` is set."""


@dataclass
class UnitTokenizationConfig(Config):
    from_model: Optional[str] = "seamlessM4T_large"
    """If set, using tokenizer from a model card."""

    num_units: Optional[int] = None
    """Alternatively, build custom tokenizer, set number of units"""

    langtoks: Optional[List[str]] = None
    """List of language tokens that should be added. Not used if `from_model` is set."""


@dataclass
class AudioProcessingConfig(Config):
    audio_root_dir: str = "/"
    """The root directory of the zipped audio files."""

    fbanks_standardize_audio: bool = True

    fbanks_num_mel_bins: int = 80

    fbanks_waveform_scale: int = 2**15


@dataclass
class DataLoadingConfig(Config):
    manifest_list_path: Optional[str] = None
    """Path to a file with the list of tsv manifests"""

    manifest_list: Optional[str] = None
    """Comma separated list of tsv manifests. Can be combined with `manifest_list_path`"""

    manifest_path_prefix: Optional[str] = None
    """Path prefix to manifest files (root directory)"""

    audio: AudioProcessingConfig = AudioProcessingConfig()
    """ Audio processing params """

    text_tokenization: TextTokenizationConfig = TextTokenizationConfig()
    """ Text tokenization params """

    unit_tokenization: UnitTokenizationConfig = UnitTokenizationConfig()
    """ Units tokenization params """

    unit_tokenizer_name: Optional[str] = "seamlessM4T_large"

    prepend_tgt_lang_tag: bool = True
    """ Prepend output text sequence with target lang token"""

    fbank_feats_pad_idx: int = 0
    """The pad index to use in fbanks batching."""

    max_tgt_text_tokens_per_batch: Optional[int] = 1000
    """ Defines flexible batch construction """

    max_batch_size: Optional[int] = None
    """ In flexible batch construction sets max allowed size"""

    fixed_batch_size: Optional[int] = None
    """ If set, uses fixed batch size """

    max_seconds_per_input_audio: int = 15
    """Accept only samples with less than max_seconds_per_input_audio ( waveform.shape[0] * SR )"""

    max_tgt_text_tokens_per_sample: int = 300
    """Accept only samples with less than max_sequence_length units"""

    max_units_per_sample: int = 1500
    """Accept only samples with less than max_sequence_length units"""

    num_threads: int = 5
    """The number of parallel threads during data reading and processing."""

    shuffle_window: Optional[int] = 1000
    """The size of sliding shuffle window."""

    prefech_batches: Optional[int] = 10
    """How many batches to prefetch in the background."""


@dataclass
class CustomModelParams(Config):
    model_embed_dim: int = 1024

    w2v2_encoder_layers: int = 24

    w2v2_encoder_layers_use_conformer: bool = True

    w2v2_encoder_layers_layernorm_features: bool = False

    w2v2_pos_encoder_type: Literal["conv", "relative", "rotary"] = "relative"

    w2v2_pos_encoder_depth: int = 0

    w2v2_pos_conv_kernel_size: int = 0

    w2v2_num_pos_conv_groups: int = 0

    nllb_encoder_layers: int = 24

    nllb_decoder_layers: int = 24

    t2u_encoder_layers: int = 6

    t2u_decoder_layers: int = 6

    nllb_vocabulary_size: int = 256102  # num_tokens + langs + spec symbols

    unit_vocabulary_size: int = 10082


@dataclass
class ModelConfig(Config):
    from_model: Optional[str] = None
    """If set, initialize a model defined in model cards. Also loads model weights."""

    from_model_config: Optional[str] = None
    """If set, initialize a model defined in model cards. Doesn't load weights."""

    custom_params: Optional[CustomModelParams] = None
    """If set, intitalize a new model with custom parameters"""

    pretrained_w2v2_path: Optional[str] = None
    """If set, use pre-trained w2v block"""

    pretrained_s2t_decoder_path: Optional[str] = None
    """If set, use pre-trained s2t decoder (NLLB)"""

    pretrained_t2u_path: Optional[str] = None
    """If set, use pre-trained t2u weights"""


@dataclass
class TrainingParams(Config):
    max_epochs: int = 100
    """ Maximum number of trainign epochs"""

    label_smoothing: float = 0.2
    """ Label smoothing coefficient for nll_loss """

    warmup_steps: int = 1000
    """ Number of steps with linearly increasing LR"""

    log_steps: int = 200
    """ Log inner loss after each `log_steps` training steps"""

    eval_steps: int = 1000
    """ Get eval loss after each `eval_steps` training steps """

    patience: int = 10
    """ Terminate if eval loss did not improve
    over the last `patience * eval_steps` training steps"""

    learning_rate: float = 1e-4
    """ Optimizer learining rate """

    start_learning_rate: float = 1e-7
    """ Start learining rate """

    float_dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
    """ Dtype used for float numbers, defines training precision """


@dataclass
class WorkflowParams(Config):
    training: TrainingParams

    model: ModelConfig

    train_data: DataLoadingConfig

    eval_data: DataLoadingConfig
