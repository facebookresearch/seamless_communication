# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.typing import DataType, Device

from seamless_communication.models.vocoder.codehifigan import CodeGenerator
from seamless_communication.models.vocoder.vocoder import Vocoder


@dataclass
class VocoderConfig:
    """Holds the configuration of a Vocoder model."""

    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    model_in_dim: int
    num_embeddings: int
    embedding_dim: int
    dur_predictor_params: Dict[str, float]
    lang_embedding_dim: int
    num_langs: int
    spkr_embedding_dim: int
    num_spkrs: int
    lang_spkr_idx_map: Dict[str, Any]


vocoder_archs = ArchitectureRegistry[VocoderConfig]("vocoder_code_hifigan")

vocoder_arch = vocoder_archs.decorator


@vocoder_arch("base")
def _base_vocoder() -> VocoderConfig:
    return VocoderConfig(
        upsample_rates=[5, 4, 4, 2, 2],
        upsample_kernel_sizes=[11, 8, 8, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        model_in_dim=1792,
        num_embeddings=10000,
        embedding_dim=1280,
        dur_predictor_params={
            "encoder_embed_dim": 1280,
            "var_pred_hidden_dim": 1280,
            "var_pred_kernel_size": 3,
            "var_pred_dropout": 0.5,
        },
        lang_embedding_dim=256,
        num_langs=36,
        spkr_embedding_dim=256,
        num_spkrs=200,
        lang_spkr_idx_map={},
    )


class VocoderBuilder:
    """Builds modules of a vocoder model (Code Hifigan) as described in
    :cite:t`https://github.com/facebookresearch/speech-resynthesis`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: VocoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: VocoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device, self.dtype = device, dtype

    def build_model(self) -> Vocoder:
        """Build a model."""

        code_generator = CodeGenerator(
            self.config.upsample_rates,
            self.config.upsample_kernel_sizes,
            self.config.upsample_initial_channel,
            self.config.resblock_kernel_sizes,
            self.config.resblock_dilation_sizes,
            self.config.model_in_dim,
            self.config.num_embeddings,
            self.config.embedding_dim,
            self.config.dur_predictor_params,
            self.config.lang_embedding_dim,
            self.config.num_langs,
            self.config.spkr_embedding_dim,
            self.config.num_spkrs,
        )
        code_generator.to(device=self.device, dtype=self.dtype)
        vocoder = Vocoder(code_generator, self.config.lang_spkr_idx_map)
        return vocoder


def create_vocoder_model(
    config: VocoderConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Vocoder:
    """Create a Vocoder model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    return VocoderBuilder(config, device=device, dtype=dtype).build_model()
