# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MultimodalSample:
    id: int
    lang: str
    text: str
    audio_local_path: Optional[str] = None
    waveform: Optional[torch.Tensor] = None
    sampling_rate: Optional[int] = None
    units: Optional[List[int]] = None

    @classmethod
    def from_json(cls, js: Dict[str, Any]) -> "MultimodalSample":
        return cls(
            id=js["id"],
            lang=js["lang"],
            text=js["text"],
            audio_local_path=js.get("audio_local_path"),
            waveform=None,  # don't serialize
            sampling_rate=js.get("sampling_rate"),
            units=js.get("units"),
        )


@dataclass
class LangPairSample:
    source: MultimodalSample
    target: MultimodalSample

    @classmethod
    def from_json(cls, js: Dict[str, Any]) -> "LangPairSample":
        return cls(
            source=MultimodalSample.from_json(js["source"]),
            target=MultimodalSample.from_json(js["target"]),
        )
