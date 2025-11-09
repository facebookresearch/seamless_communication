# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

__version__ = "0.1.0"


from pathlib import Path
# from fairseq2 import init_fairseq2
# from fairseq2.runtime.dependency import DependencyContainer
# from fairseq2.composition import register_file_assets

# def _setup_assets(container: DependencyContainer) -> None:
#     cards_dir = Path(__file__).parent / "cards"
#     register_file_assets(container, cards_dir)

# # initialise fairseq2 et enregistre tes cards
# init_fairseq2(extras=_setup_assets)


# from pathlib import Path

from fairseq2.runtime.dependency import DependencyContainer
# from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.composition import register_file_assets, register_model_family

from seamless_communication.models.aligner.builder import UnitY2AlignmentConfig, create_unity2_alignment_model, load_arch_unity2_aligner
from seamless_communication.models.aligner.model import UnitY2AlignmentModel
from seamless_communication.models.generator.builder import load_arch_vocoder_pretssel, VocoderConfig, create_vocoder_model, PretsselVocoder
from seamless_communication.models.monotonic_decoder.builder import (
    MonotonicDecoderConfig, MonotonicDecoderModel, create_monotonic_decoder_model, load_arch_monotonic
)

from seamless_communication.models.unity.builder import (
    UnitYConfig, UnitYModel, create_unity_model, load_arch_unity
)

from seamless_communication.models.conformer_shaw.builder import (
    ConformerShawEncoderConfig, load_arch_conformer_shaw, create_conformer_shaw_model, Wav2Vec2Model, ConformerShawEncoderFactory
)

from seamless_communication.models.unity.loader import convert_unity_checkpoint
from seamless_communication.models.unity.model import UnitYNART2UModel, UnitYT2UModel
from seamless_communication.models.unity.t2u_builder import (
    create_unity_t2u_model, load_arch_unity_t2u, UnitYT2UConfig
)
from seamless_communication.toxicity.mutox.classifier import MutoxClassifier, MutoxConfig
from seamless_communication.toxicity.mutox.builder import create_mutox_model, load_arch_mutox_classifier

def setup_seamless_extension(container: DependencyContainer) -> None:
    # (facultatif) si tu as des YAML d’assets dans ton package
    # register_file_assets(container, Path(__file__).parent / "assets")

    # Remplaçant de ArchitectureRegistry[...]("monotonic_decoder")
    cards_dir = Path(__file__).parent / "cards"
    register_file_assets(container, cards_dir)

    # 2) la famille modèle
    register_model_family(
        container,
        "monotonic_decoder",                    # nom de la famille
        kls=MonotonicDecoderModel,              # classe de modèle
        config_kls=MonotonicDecoderConfig,      # classe de config
        factory=create_monotonic_decoder_model, # fabrique (poids aléatoires)
    )
    # 3) les architectures (presets) via un registrar
    load_arch_monotonic(container)

    # 2) la famille modèle
    register_model_family(
        container,
        "unity",                    # nom de la famille
        kls=UnitYModel,              # classe de modèle
        config_kls=UnitYConfig,      # classe de config
        factory=create_unity_model, # fabrique (poids aléatoires)
        state_dict_converter=convert_unity_checkpoint
    )

    load_arch_unity(container)

    # 2) la famille modèle
    register_model_family(
        container,
        "conformer_shaw",                    # nom de la famille
        kls=Wav2Vec2Model,              # classe de modèle
        config_kls=ConformerShawEncoderConfig,      # classe de config
        factory=create_conformer_shaw_model, # fabrique (poids aléatoires)
    )

    load_arch_conformer_shaw(container)

    register_model_family(
        container,
        "unity_t2u",                    # nom de la famille
        kls=UnitYT2UModel,              # classe de modèle
        config_kls=UnitYT2UConfig,      # classe de config
        factory=create_unity_t2u_model, # fabrique (poids aléatoires)
    )
    load_arch_unity_t2u(container)

    # register_model_family(
    #     container,
    #     "unity_nart2u",                    # nom de la famille
    #     kls=UnitYNART2UModel,              # classe de modèle
    #     config_kls=UnitYT2UConfig,      # classe de config
    #     factory=create_unity_nart2u_model, # fabrique (poids aléatoires)
    # )

    register_model_family(
        container,
        "vocoder_pretssel",                    # nom de la famille
        kls=PretsselVocoder,              # classe de modèle
        config_kls=VocoderConfig,      # classe de config
        factory=create_vocoder_model, # fabrique (poids aléatoires)
    )

    load_arch_vocoder_pretssel(container)

    register_model_family(
        container,
        "mutox_classifier",
        kls=MutoxClassifier,
        config_kls=MutoxConfig,
        factory=create_mutox_model,
    )

    load_arch_mutox_classifier(container)

    register_model_family(
        container,
        "unity2_alignment",
        kls=UnitY2AlignmentModel,
        config_kls=UnitY2AlignmentConfig,
        factory=create_unity2_alignment_model,
    )

    load_arch_unity2_aligner(container)
