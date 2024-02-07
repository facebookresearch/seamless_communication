"""
Fine Tune Timestamp Extraction
Load transcriptions pre-made with Whisper, run audio through Seamless and compare timestamps.
It is expected that the files `transcriptions/whisper_{lang}.json` contain data in the format:
```json
[
    {
        "audio_path": "/path/to/file.wav",
        "text": "transcribed text",
        "words": [
            {
                "word": "transcribed",
                "start": 0.1,
                "end": 0.25,
                "probability": 0.8
            },
            ...
        ]
    },
    ...
]
```
"""
from itertools import product
from pathlib import Path
from typing import Optional


import torch
from fairseq2.data import VocabularyInfo
from fairseq2.models.nllb.builder import NllbConfig
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.nn.transformer import TransformerNormOrder
from seamless_communication.models.tokenizer import SPMTokenizer
from seamless_communication.models.unity import (
    UnitYModel,
    UnitYConfig,
    UnitYT2UConfig,
    create_unity_model,
    load_unity_unit_tokenizer,
)


def _build_vocab_info(
    vocab_size: int, ref_vocab_info: Optional[VocabularyInfo]
) -> VocabularyInfo:
    assert ref_vocab_info is not None
    return VocabularyInfo(
        size=vocab_size,
        unk_idx=ref_vocab_info.unk_idx,
        bos_idx=ref_vocab_info.bos_idx,
        eos_idx=ref_vocab_info.eos_idx,
        pad_idx=ref_vocab_info.pad_idx,
    )


def _small_model_config(
    text_vocab_info: VocabularyInfo,
    unit_vocab_info: VocabularyInfo,
    model_embed_dim: int = 256,
    num_fbank_channels: int = 80,
    fbank_stride: int = 2,
) -> UnitYConfig:
    nllb_ffn_inner_dim = model_embed_dim * 8
    w2v2_ffn_inner_dim = model_embed_dim * 4
    nllb_decoder_layers = 3
    nllb_encoder_layers = 1
    nllb_vocabulary_size = 20010
    t2u_decoder_layers = 1
    t2u_encoder_layers = 1
    unit_vocabulary_size = 10082
    w2v2_encoder_layers = 6
    w2v2_encoder_layers_layernorm_features = False
    w2v2_encoder_layers_use_conformer = True
    w2v2_num_pos_conv_groups = 16
    w2v2_pos_conv_kernel_size = 128
    w2v2_pos_encoder_depth = 1
    w2v2_pos_encoder_type = "relative"
    return UnitYConfig(
        use_gelu=False,
        use_text_decoder=True,
        prosody_encoder_config=None,
        model_dim=model_embed_dim,
        w2v2_encoder_config=Wav2Vec2EncoderConfig(
            model_dim=model_embed_dim,
            max_seq_len=4096,
            feature_dim=num_fbank_channels * fbank_stride,
            use_fbank=True,
            first_pass_dropout_p=0.0,
            layer_norm_features=w2v2_encoder_layers_layernorm_features,
            feature_extractor_layer_descs=[],
            feature_extractor_bias=False,
            feature_extractor_layer_norm_convs=False,
            feature_grad_scale=0,
            num_fbank_channels=num_fbank_channels,
            fbank_stride=fbank_stride,
            sample_fbank_every_k=1,
            pos_encoder_type=w2v2_pos_encoder_type,
            pos_encoder_depth=w2v2_pos_encoder_depth,
            pos_conv_kernel_size=w2v2_pos_conv_kernel_size,
            num_pos_conv_groups=w2v2_num_pos_conv_groups,
            use_conformer=w2v2_encoder_layers_use_conformer,
            num_encoder_layers=w2v2_encoder_layers,
            num_encoder_attn_heads=16,
            ffn_inner_dim=w2v2_ffn_inner_dim,
            dropout_p=0.0,
            attn_dropout_p=0.0,
            layer_drop_p=0.0,
            norm_order=TransformerNormOrder.POST,
            depthwise_conv_kernel_size=31,
        ),
        mt_model_config=NllbConfig(
            model_dim=model_embed_dim,
            max_seq_len=1024,
            vocab_info=_build_vocab_info(
                vocab_size=nllb_vocabulary_size, ref_vocab_info=text_vocab_info
            ),
            num_encoder_layers=nllb_encoder_layers,
            num_decoder_layers=nllb_decoder_layers,
            num_encoder_attn_heads=16,
            num_decoder_attn_heads=16,
            ffn_inner_dim=nllb_ffn_inner_dim,
            dropout_p=0.1,
        ),
        t2u_config=UnitYT2UConfig(
            use_gelu=False,
            char_pad_idx=0,
            use_prosody_proj=False,
            prosody_encoder_dim=0,
            model_dim=model_embed_dim,
            unit_max_seq_len=2048,
            target_vocab_info=_build_vocab_info(
                vocab_size=unit_vocabulary_size, ref_vocab_info=unit_vocab_info
            ),
            num_encoder_layers=t2u_encoder_layers,
            num_decoder_layers=t2u_decoder_layers,
            nar_decoder_frontend_config=None,
            nar_decoder_config=None,
            num_encoder_attn_heads=16,
            num_decoder_attn_heads=16,
            ffn_inner_dim=model_embed_dim * 8,
            dropout_p=0.1,
        ),
        use_text_encoder=True,
        use_conformer_adaptor=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


def load_small_model(checkpoint_path: Path, spm_model_path: Path):
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    float_dtype = torch.float32

    unit_tokenizer = load_unity_unit_tokenizer("seamlessM4T_large")
    # unit_tokenizer.vocab_info.size = 20010
    text_tokenizer = SPMTokenizer(
        pathname=spm_model_path,
        langs=sorted(["eng", "rus", "hin", "por", "spa"]),
    )
    model_config = _small_model_config(
        text_vocab_info=text_tokenizer.vocab_info,
        unit_vocab_info=unit_tokenizer.vocab_info,
    )
    model = create_unity_model(config=model_config, dtype=float_dtype, device=device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    # print(model)
    return model, text_tokenizer


class NanoModel:
    @staticmethod
    def get_nano_model() -> tuple[UnitYModel, SPMTokenizer]:
        return load_small_model(
            checkpoint_path="nano/nano_m4t.pt", spm_model_path="nano/nano_spm.model"
        )


###

from datetime import datetime
import json

from fine_tune_transcriber import FineTuneTranscriber
from seamless_communication.inference import Transcriber

WHISPER_TO_SEAMLESS = {
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "ru": "rus",
}
INPUT_PATH = "transcriptions"
OUTPUT_PATH = "transcriptions/2algo_3filt_3wid_2rerun_50samp"
MODEL_NAME = "seamlessM4T_v2_large"
MODEL_NANO = Transcriber(NanoModel.get_nano_model())
MODEL_LARGE = Transcriber(MODEL_NAME)
# Load text pre-transcribed with Whisper
transcriptions_nano = list()
transcriptions_large = list()
for w_lang, s_lang in WHISPER_TO_SEAMLESS.items():
    with open(f"{INPUT_PATH}/whisper_{w_lang}.json", "r", encoding="utf-16") as file:
        current_transcriptions = json.loads(file.read())
        for transcription in current_transcriptions[:50]:
            transcription["lang"] = s_lang
            transcriptions_large.append(transcription)
            if s_lang in ["eng", "spa", "rus"]:
                transcriptions_nano.append(transcription)

ftt_nano = FineTuneTranscriber(MODEL_NANO, transcriptions_nano)
ftt_large = FineTuneTranscriber(MODEL_LARGE, transcriptions_large)

for algo, filter_t, width, rerun in list(
    product(
        # ["LIS", "DTW"],  # algorithm
        ["DTW"],  # algorithm
        ["", "median", "gaussian"],  # filter type
        [0, 3, 5],  # filter width
        [False, True],  # re-run decoder
    )
):
    if filter_t == "" and width > 0:  # skip diff filt widths if no filter
        continue
    if filter_t in ["gaussian", "median"] and width == 0:  # skip 0-width filters
        continue
    use_dtw = algo == "DTW"
    print(
        f"algorithm: {algo}",
        f"filter_type: {filter_t if filter_t else None}",
        f"filter_width: {width}",
        f"rerun_decoder: {rerun}",
        sep=", ",
    )
    print("nano")
    results_nano = ftt_nano.compare(
        filter_width=width,
        filter_type=filter_t,
        use_dtw=use_dtw,
        rerun_decoder=rerun,
    )
    results_nano["metadata"] = {
        "model_name": "nano",
        "algorithm": algo,
        "filter_width": width,
        "filter_type": filter_t if filter_t else None,
        "rerun_decoder": rerun,
    }
    with open(
        f"{OUTPUT_PATH}/results_{int(datetime.now().timestamp())}.json",
        "w",
        encoding="utf-16",
    ) as file:
        file.write(json.dumps(results_nano, indent=2, ensure_ascii=False))

    print("v2_large")
    results_large = ftt_large.compare(
        filter_width=width,
        filter_type=filter_t,
        use_dtw=use_dtw,
        rerun_decoder=rerun,
    )
    results_large["metadata"] = {
        "model_name": "v2_large",
        "algorithm": algo,
        "filter_width": width,
        "filter_type": filter_t if filter_t else None,
        "rerun_decoder": rerun,
    }
    with open(
        f"{OUTPUT_PATH}/results_{int(datetime.now().timestamp())}.json",
        "w",
        encoding="utf-16",
    ) as file:
        file.write(json.dumps(results_large, indent=2, ensure_ascii=False))
