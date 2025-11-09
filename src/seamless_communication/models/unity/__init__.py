# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from fairseq2.models.hub import ModelHub
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.assets import get_asset_store

from .builder import UnitYT2UConfig
from .model import UnitYT2UModel, UnitYNART2UModel 

from seamless_communication.models.unity.builder import UnitYBuilder as UnitYBuilder
from seamless_communication.models.unity.builder import UnitYConfig as UnitYConfig
from seamless_communication.models.unity.builder import (
    create_unity_model as create_unity_model,
)
# from seamless_communication.models.unity.builder import unity_arch as unity_arch
# from seamless_communication.models.unity.builder import unity_archs as unity_archs
from seamless_communication.models.unity.char_tokenizer import (
    CharTokenizer as CharTokenizer,
)
from seamless_communication.models.unity.char_tokenizer import (
    UnitYCharTokenizerLoader as UnitYCharTokenizerLoader,
)
from seamless_communication.models.unity.char_tokenizer import (
    load_unity_char_tokenizer as load_unity_char_tokenizer,
)
from seamless_communication.models.unity.fft_decoder import (
    FeedForwardTransformer as FeedForwardTransformer,
)
from seamless_communication.models.unity.fft_decoder_layer import (
    FeedForwardTransformerLayer as FeedForwardTransformerLayer,
)
from seamless_communication.models.unity.film import FiLM
from seamless_communication.models.unity.length_regulator import (
    HardUpsampling as HardUpsampling,
)
from seamless_communication.models.unity.length_regulator import (
    VarianceAdaptor as VarianceAdaptor,
)
from seamless_communication.models.unity.length_regulator import (
    VariancePredictor as VariancePredictor,
)
from seamless_communication.models.unity.loader import (
    convert_unity_checkpoint,
    load_gcmvn_stats as load_gcmvn_stats,
)
from seamless_communication.models.unity.loader import (
    load_unity_t2u_config as load_unity_t2u_config,
    load_unity_config as load_unity_config
    # load_unity_nart2u_config as load_unity_nart2u_config
)

from seamless_communication.models.unity.loader import (
    load_unity_text_tokenizer as load_unity_text_tokenizer,
)
from seamless_communication.models.unity.loader import (
    load_unity_unit_tokenizer as load_unity_unit_tokenizer,
)
from seamless_communication.models.unity.model import UnitYModel as UnitYModel
from seamless_communication.models.unity.model import (
    UnitYNART2UModel as UnitYNART2UModel,
)
from seamless_communication.models.unity.model import UnitYOutput as UnitYOutput
from seamless_communication.models.unity.model import UnitYT2UModel as UnitYT2UModel
from seamless_communication.models.unity.model import UnitYX2TModel as UnitYX2TModel
from seamless_communication.models.unity.nar_decoder_frontend import (
    NARDecoderFrontend as NARDecoderFrontend,
)
from seamless_communication.models.unity.t2u_builder import (
    UnitYT2UConfig as UnitYT2UConfig,
)
from seamless_communication.models.unity.t2u_builder import (
    create_unity_t2u_model as create_unity_t2u_model,
    # get_unity_t2u_model_hub as get_unity_t2u_model_hub
)
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenDecoder as UnitTokenDecoder,
)
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenEncoder as UnitTokenEncoder,
)
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenizer as UnitTokenizer,
)

_FAMILY = "unity"

def get_unity_model_hub() -> ModelHub[UnitYModel, UnitYConfig]:
    resolver = get_dependency_resolver()
    family = resolver.resolve(ModelFamily, key=_FAMILY)
    return ModelHub(family, get_asset_store())

_FAMILY_2 = "unity_t2u"

def get_unity_t2u_model_hub() -> ModelHub[UnitYT2UModel, UnitYT2UConfig]:
    resolver = get_dependency_resolver()
    family = resolver.resolve(ModelFamily, key=_FAMILY_2)
    return ModelHub(family, get_asset_store())


# _FAMILY_2 = "unity_nart2u"
# def get_unity_nart2u_model_hub() -> ModelHub[UnitYNART2UModel, UnitYT2UConfig]:
#     resolver = get_dependency_resolver()
#     family = resolver.resolve(ModelFamily, key=_FAMILY_2)
#     return ModelHub(family, get_asset_store())


# load_unity_config = ConfigLoader[UnitYConfig](asset_store, unity_archs)


# load_unity_model = ModelLoader[UnitYModel, UnitYConfig](
#     asset_store,
#     download_manager,
#     load_unity_config,
#     create_unity_model,
#     convert_unity_checkpoint,
#     restrict_checkpoints=False,
# )
# def load_unity_model(
#     name: str,
#     *,
#     device = None,
#     dtype = None,
# ) :
#     return get_unity_model_hub().load_model(name, device=device, dtype=dtype)

# from fairseq2.model_checkpoint import DelegatingModelCheckpointLoader

def load_unity_model(name_or_card: str, *, config=None,device=None, dtype=None) -> UnitYModel:
    return get_unity_model_hub().load_model(name_or_card, device=device, dtype=dtype, config=config)

# def load_unity_model(name: str, *, device=None, dtype=None):
#     hub = get_unity_model_hub()

#     # 1) Récupère la config via le hub
#     cfg = hub.get_model_config(name)

#     # 2) Instancie le modèle vide
#     model = create_unity_model(cfg, device=device, dtype=dtype)

#     # 3) Résout le checkpoint depuis la card (resource 'checkpoint')
#     card = get_asset_store().retrieve_card(name)
#     ckpt_uri = card.field("checkpoint").as_(str)
#     # ckpt_uri = next(res["uri"] for res in resources if isinstance(res, dict) and res.get("name") == "checkpoint")

#     # 4) Charge + convertit l’état
#     loader = DelegatingModelCheckpointLoader()
#     state = {
#         k: t
#         for k, t in loader.lazy_load(
#             ckpt_uri,
#             state_dict_converter=lambda s: convert_unity_checkpoint(s, cfg),
#         )
#     }

#     # 5) Applique au modèle
#     model.load_state_dict(state, strict=False)
#     return model

def load_unity_t2u_model(
    name: str,
    *,
    device = None,
    dtype = None,
) :
    return get_unity_t2u_model_hub().load_model(name, device=device, dtype=dtype)
