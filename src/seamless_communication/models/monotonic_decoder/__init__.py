# seamless_communication/models/monotonic_decoder/__init__.py

# --- ré-export propres -------------------------------------------------------
from .builder import (
    MonotonicDecoderBuilder,
    MonotonicDecoderConfig,
    create_monotonic_decoder_model,
)
from .model import MonotonicDecoderModel
from .loader import (
    load_monotonic_decoder_config,
    load_monotonic_decoder_model,
)

# --- hub pour la famille -----------------------------------------------------
from typing import TypeVar, Generic
from fairseq2.assets import get_asset_store
from fairseq2.models.hub import ModelHub
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver

_FAMILY_NAME = "monotonic_decoder"

def get_monotonic_decoder_model_hub() -> ModelHub[MonotonicDecoderModel, MonotonicDecoderConfig]:
    """
    Renvoie le hub pour la famille 'monotonic_decoder'.
    Suppose que la famille et ses archs ont été enregistrées via init_fairseq2(extras=...).
    """
    resolver = get_dependency_resolver()  # <- remplace get_global_container()
    family = resolver.resolve(ModelFamily, key=_FAMILY_NAME)
    return ModelHub(family, get_asset_store())
