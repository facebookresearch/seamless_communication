# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# The rules to blend the p2v decoder, mel-vocoder and the watermarking:
#
# Step 1) Make the big sequential module `layers` that consists of:
#    - PostNet (last layer of the p2v decoder) : 5 layers
#    - mel-vocoder layers (conv_pre, ups, resblocks, conv_post): 18 layers
#    - watermarking encoder and decoder: 32 layers
#
# Step 2) Take the last 32 layers of the watermarking, split into 4 blocks of
# 8 layers. Mix these blocks into the previous layers
#
# The final mixed architecture SPVM (Spaghetti Pretssel Vovoder Model):
#
#     <P2V: Post Net>
#           |
# <Block 1 of Watermarker> ------
#           |                   |
#          \/                   |
#  <Melvocoder: Conv_pre>       |
#           | (skip)            |
# <Block 2 of Watermarker> -----|
#           |                   |
#          \/                   |
# <Melvocoder: Upsampler>       |
#           | (skip)            |
# <Block 3 of Watermarker> -----|
#           |                   |
#          \/                   |
# <Melvocoder: Resblocks>       |
#           | (skip)            |
# <Block 4 of Watermarker> -----|
#           |                   |
#          \/                   |
#  <Melvocoder: Conv_post>      |
#           |                   |
#           | ------------------|
#           |
#          \/
#    watermarked wavs

from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Mapping, Match

import torch
from fairseq2.models.utils.checkpoint import (
    convert_fairseq_checkpoint,
    convert_model_state_dict,
    load_checkpoint,
)


def pretssel_key_map() -> Mapping[str, str]:
    """
    The rule for renaming the layers of Pretssel model checkpoint:
        - Merge decoder.postnet into `layers`
    """
    from seamless_communication.models.pretssel.loader import _fairseq_key_map  # noqa

    key_map = _fairseq_key_map(None)  # type: ignore[arg-type]
    del key_map[r"^decoder\.postnet\."]
    key_map[r"^decoder\.postnet\.convolutions\."] = r"layers."
    return key_map


def vocoder_key_map() -> Mapping[str, Any]:
    """
    Rename layers in the mel-vocoder checkpoint. We flatten the vocoder arch and put everything
    into the `layers`, in which `postnet_size` layers from the PostNet already present. In other words:
        - conv_pre -> layers.<postnet_size + watermark / 4>
        - ups.i -> layers.<postnet_size + 1 + i + watermark_size / 2>
        - resblocks.i -> layers.<postnet_size + 1 + ups_size + i + 3 * watermark_size / 4>
        - conv_post.i -> layers.<postnet_size + 1 + ups_size + resblocks_size + i + watermark_size>
    """

    return {
        # fmt: off
        # postnet_size = 5, 1st wm block = 8 -> 13
        r"^conv_pre\.":               r"layers.13.",                                 # noqa, 2nd wm block = 8 -> +8
        r"^ups\.([0-9]+)\.":          lambda x: f"layers.{int(x.group(1)) + 22}.",   # noqa, ups_size = 4, 3rd wm block = 8 -> +12
        r"^resblocks\.([0-9]+)\.":    lambda x: f"layers.{int(x.group(1)) + 34}.",   # noqa, resblocks_size = 12, 4th wm block = 8 -> +20
        r"^conv_post\.":              r"layers.54.",
        # fmt: on
    }


def wm_key_map() -> Mapping[Any, Any]:
    """
    flatten all encoders and decoders into the one sequential layer (step 1) and split the watermaker
    into 4 blocks and mix into the layers of the p2v decoder and mel-vocoder:
        - encoder.model.[0-7] --> layers.<postnet_size + i> (5 --> 12)
        - encoder.model.[8-15] --> layers.<postnet_size + 9> (14 --> 21)
        - decoder.model.[0-7] --> layers.<postnet_size + vocoder_ups_size + conv_pre + 16> (26 -> 33)
        - decoder.model.[8-15] --> layers.<postnet_size + vocoder_ups_size + conv_pre + resblock_size + 24> (46 -> 53)
    """

    def encoder_layer_index(match_obj: Match[str]) -> str:
        idx = int(match_obj.group(1))
        # First half of the encoder is after the PostNet
        if idx < 8:
            # postnet_size = 5
            return f"layers.{idx + 5}."

        # Second half of the encoder goes after the mel-vocoder:conv_pre
        else:
            # postnet = 5, conv_pre = 1 --> +6
            return f"layers.{idx + 6}."

    def decoder_layer_index(match_obj: Match[str]) -> str:
        idx = int(match_obj.group(1))
        # First half of the decoder is after the mel-vocoder:ups
        if idx < 8:
            # postnet 5, conv_pre 1, encoder 16, ups 4 --> +26
            return f"layers.{idx + 26}."
        else:
            # postnet 5, conv_pre 1, encoder 16, ups 4, resblock 12 -> +38
            return f"layers.{idx + 38}."

    return {
        r"^encoder\.model\.([0-9]+)\.": encoder_layer_index,
        r"^decoder\.model\.([0-9]+)\.": decoder_layer_index,
    }


def combine_chkpts(
    pretssel_file: str, vocoder_file: str, wm_file: str, out_path: str
) -> None:
    """Combine the pretssel and melhifigan into one model"""
    pretssel_chkpt = load_checkpoint(pretssel_file)
    pretssel_chkpt = convert_fairseq_checkpoint(pretssel_chkpt, pretssel_key_map())

    vocoder_chkpt = load_checkpoint(vocoder_file)
    vocoder_chkpt = convert_fairseq_checkpoint(vocoder_chkpt, vocoder_key_map())

    wm_ckpt = load_checkpoint(wm_file)
    # some wm checkpoints are not a fairseq2 checkpoint, so we have to inspect it differently
    if "model" in wm_ckpt:
        wm_ckpt = wm_ckpt["model"]
    wm_ckpt = convert_model_state_dict(wm_ckpt, wm_key_map())

    # Merge the state dicts
    ckpt = pretssel_chkpt
    state_dict = ckpt["model"]
    for vocoder_key in vocoder_chkpt["model"]:
        state_dict[vocoder_key] = vocoder_chkpt["model"][vocoder_key]

    for wm_key, wm_val in wm_ckpt.items():
        if wm_key.startswith("layers."):
            state_dict[wm_key] = wm_val

    # Remove obsolete layers
    keys_to_delete = [
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
        "enc_emb_proj.weight",
        "enc_emb_proj.bias",
    ]
    keys_to_delete.extend(
        [
            key
            for key in state_dict
            if key.startswith("decoder.var_adaptor.duration_predictor")
        ]
    )
    for key in keys_to_delete:
        if key in state_dict:
            del state_dict[key]

    model_mapping_metafile = Path(out_path).with_suffix(".arch")
    with open(model_mapping_metafile, "w", encoding="utf-8") as o:
        o.write(vocoder_key_map.__doc__)  # type: ignore
        o.write("\n")
        o.write(wm_key_map.__doc__)  # type: ignore
        o.write("\n")
    torch.save(ckpt, out_path)


if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser(description="Compile watermarking into p2v decoder and vocoder")
    parser.add_argument(
        "--pretssel",
        default="/checkpoint/mjhwang/experiments/230930-noiseaug_p2v-mls_multilingual_6lang/231005-noiseaug_p2v-mls_multilingual_6lang-alignfix.config_v2.langemb1.vuv_logit1.denoise.ngpu16/checkpoint_best.pt",
        type=str,
        help="Path to the Pretssel model checkpoint",
    )
    parser.add_argument(
        "--vocoder",
        # default="/large_experiments/seamless/ust/changhan/checkpoints/fairseq2/pretssel_hifigan.pt",
        default="/large_experiments/seamless/workstream/expressivity/oss/checkpoints/melhifigan_20231121.pt",
        type=str,
        help="Path to the mel-vocoder checkpoint",
    )
    parser.add_argument(
        "--wm",
        default="/checkpoint/hadyelsahar/experiments/audiocraft/outputs/xps/BA6f05be46/checkpoint.th",
        type=str,
        help=""
    )
    parser.add_argument(
        "--output",
        default="/large_experiments/seamless/workstream/expressivity/oss/checkpoints/pretssel_melhifigan_wm-final.pt",
        # default="/large_experiments/seamless/workstream/expressivity/oss/checkpoints/pretssel_melhifigan_wm-20231121.pt",
        type=str,
        help="Path to the output watermarked model checkpoint",
    )
    # fmt: on
    args = parser.parse_args()
    combine_chkpts(args.pretssel, args.vocoder, args.wm, args.output)
