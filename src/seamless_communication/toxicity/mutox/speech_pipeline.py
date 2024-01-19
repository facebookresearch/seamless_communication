# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import torch
from seamless_communication.toxicity.mutox.classifier import MutoxClassifier
from seamless_communication.toxicity.mutox.loader import load_mutox_model
from sonar.models.sonar_speech.loader import load_sonar_speech_model

from sonar.inference_pipelines.speech import (
    SpeechToEmbeddingPipeline,
    SpeechInferenceParams,
)

from fairseq2.data import (
    DataPipelineBuilder,
)

from typing import Union

from seamless_communication.toxicity.mutox.classifier import MutoxClassifier
from sonar.models.encoder_model import SonarEncoderModel
from fairseq2.typing import Device


CPU_DEVICE = torch.device("cpu")


class MutoxSpeechClassifierPipeline(SpeechToEmbeddingPipeline):
    def __init__(
        self,
        mutox_classifier: Union[str, MutoxClassifier],
        encoder: Union[str, SonarEncoderModel],
        device: Device = CPU_DEVICE,
    ) -> None:
        super().__init__(encoder)
        self.model.to(device).eval()
        self.mutox_classifier = mutox_classifier.to(device).eval()

    @classmethod
    def load_model_from_name(
        cls,
        mutox_classifier_name: str,
        encoder_name: str,
        device: Device = CPU_DEVICE,
    ) -> "SpeechToEmbeddingPipeline":
        encoder = load_sonar_speech_model(encoder_name, device=device, progress=False)
        mutox_classifier = load_mutox_model(
            mutox_classifier_name, device=device, progress=False
        )
        return cls(mutox_classifier=mutox_classifier, encoder=encoder, device=device)

    def prebuild_pipeline(self, context: SpeechInferenceParams) -> DataPipelineBuilder:
        pipeline_builder = super().prebuild_pipeline(context)
        return pipeline_builder.map(self._run_classifier, selector="audio.data")

    @torch.inference_mode()
    def _run_classifier(self, data: dict):
        return self.mutox_classifier(data.sentence_embeddings)
