# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


from fairseq2.models import setup_model_family

from seamless_communication.models.generator.vocoder import PRETSSEL_VOCODER_FAMILY
from seamless_communication.models.generator.builder import (
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
)

load_pretssel_vocoder_model, load_pretssel_vocoder_config = setup_model_family(
    PRETSSEL_VOCODER_FAMILY,
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
    restrict_checkpoints=False,
)
