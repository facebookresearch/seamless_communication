# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# The original implementation for the watermarking
# This is not open-sourced and only kept here for future reference
# mypy: ignore-errors

import math
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
from typing import Any, Dict, Union

import audiocraft
import omegaconf
import torch
import torch.nn as nn
import torchaudio
from audiocraft.modules.seanet import SEANetEncoder
from audiocraft.utils.utils import dict_from_config
from fairseq2.typing import DataType, Device


class SEANetEncoderKeepDimension(SEANetEncoder):
    """
    similar architecture to the SEANet encoder but with an extra step that
    projects the output dimension to the same input dimension by repeating
    the sequential

    Args:
        SEANetEncoder (_type_): _description_
    """

    def __init__(self, output_hidden_dim: int = 8, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.output_hidden_dim = output_hidden_dim
        # Adding a reverse convolution layer
        self.reverse_convolution = nn.ConvTranspose1d(
            in_channels=self.dimension,
            out_channels=self.output_hidden_dim,
            kernel_size=math.prod(self.ratios),
            stride=math.prod(self.ratios),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_nframes = x.shape[-1]
        x = self.model(x)
        x = self.reverse_convolution(x)
        # make sure dim didn't change
        x = x[:, :, :orig_nframes]
        return x


class Watermarker(nn.Module):
    """
    Initialize the Watermarker model.

    Args:
        encoder (nn.Module): Watermark Encoder.
        decoder (nn.Module): Watermark Decoder.
        detector (nn.Module): Watermark Detector.
    """

    encoder: SEANetEncoder
    decoder: SEANetEncoder
    detector: SEANetEncoderKeepDimension

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetEncoder,
        detector: SEANetEncoderKeepDimension,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.detector = detector

    def get_watermark(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the watermark from a batch of audio input.

        Args:
            x (torch.Tensor): Input audio tensor with dimensions [batch size, channels = 1, frames].

        Returns:
            torch.Tensor: Output watermark with the same dimensionality as the input.
        """
        hidden = self.encoder(x)
        # assert dim in = dim out
        watermark = self.decoder(hidden)[:, :, : x.size(-1)]
        return watermark

    def detect_watermark(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect the watermark in a batch of audio input.

        Args:
            x (torch.Tensor): Input audio tensor with dimensions
            [batch size, channels = 1, frames].

        Returns:
            torch.Tensor: Predictions of the classifier for watermark
            with dimensions [bsz, classes = 2, frames].
            For each frame, the detector outputs probabilities of
            non-watermarked class (class id 0) and
            the probability of "watermarked" class (class id 1).
            To do inference, you can use output[:, 1, :]
            to get probabilities of input audio being watermarked.
        """
        return self.detector(x)


def model_from_checkpoint(
    config_file: Union[Path, str] = "seamlesswatermark.yaml",
    checkpoint: str = "",
    device: Union[torch.device, str] = "cpu",
    dtype: DataType = torch.float32,
) -> Watermarker:
    """Instantiate a Watermarker model from a given checkpoint path.

    Example usage:
    >>> from watermarking.watermarking import *
    >>> cfg = "seamlesswatermark.yaml"
    >>> url = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
    >>> urllib.request.urlretrieve(url, "random.wav")
    >>> wav, _ = torchaudio.load("random.wav")
    >>> wav = wav.unsqueeze(0)  # add bsz dimension

    >>> model = model_from_config(cfg, device = wav.device)
    # Other way is to load directly from the checkpoint
    >>> model = model_from_checkpoint(checkpoint_path, device = wav.device)

    >>> watermark = model.get_watermark(wav)

    >>> watermarked_audio = wav + watermark
    >>> detection = model.detect_watermark(watermarked_audio)
    >>> print(detection[:,1,:])  # print prob of watermarked class # should be > 0.5

    >>> detection = model.detect_watermark(wav)
    >>> print(detection[:,1,:])  # print prob of watermarked class  # should be < 0.5

    Args:
        checkpoint_path (Path or str): Path to the checkpoint file.
        device (torch.device or str, optional): Device on which
        the model is loaded (default is "cpu").

    Returns:
        Watermarker: An instance of the Watermarker model loaded from the checkpoint.
    """
    config_path = Path(__file__).parent / config_file
    cfg = omegaconf.OmegaConf.load(config_path)
    if checkpoint and Path(checkpoint).is_file():
        ckpt = checkpoint
    else:
        ckpt = cfg["checkpoint"]
    state: Dict[str, Any] = torch.load(ckpt, map_location=device)
    if "model" in state and "xp.cfg" in state:
        cfg = omegaconf.OmegaConf.create(state["xp.cfg"])
        omegaconf.OmegaConf.resolve(cfg)
        state = state["model"]
    watermarking_model = get_watermarking_model(cfg)
    watermarking_model.load_state_dict(state)
    watermarking_model = watermarking_model.to(device, dtype=dtype)
    watermarking_model.eval()
    return watermarking_model


def get_watermarking_model(cfg: omegaconf.DictConfig) -> Watermarker:
    encoder, decoder = get_encodec_autoencoder(cfg)
    detector = get_detector(cfg)
    return Watermarker(encoder, decoder, detector)


def get_encodec_autoencoder(cfg: omegaconf.DictConfig):
    kwargs = dict_from_config(getattr(cfg, "seanet"))
    if hasattr(cfg.seanet, "detector"):
        kwargs.pop("detector")
    encoder_override_kwargs = kwargs.pop("encoder")
    decoder_override_kwargs = kwargs.pop("decoder")
    encoder_kwargs = {**kwargs, **encoder_override_kwargs}
    decoder_kwargs = {**kwargs, **decoder_override_kwargs}
    encoder = audiocraft.modules.SEANetEncoder(**encoder_kwargs)
    decoder = audiocraft.modules.SEANetDecoder(**decoder_kwargs)
    return encoder, decoder


def get_detector(cfg: omegaconf.DictConfig):
    kwargs = dict_from_config(getattr(cfg, "seanet"))
    encoder_override_kwargs = kwargs.pop("detector")
    kwargs.pop("decoder")
    kwargs.pop("encoder")
    encoder_kwargs = {**kwargs, **encoder_override_kwargs}

    # Some new checkpoints of watermarking was trained on a newer code, where
    # `output_hidden_dim` is renamed to `output_dim`
    if "output_dim" in encoder_kwargs:
        output_hidden_dim = encoder_kwargs.pop("output_dim")
    else:
        output_hidden_dim = 8
    encoder = SEANetEncoderKeepDimension(output_hidden_dim, **encoder_kwargs)

    last_layer = torch.nn.Conv1d(output_hidden_dim, 2, 1)
    softmax = torch.nn.Softmax(dim=1)
    detector = torch.nn.Sequential(encoder, last_layer, softmax)
    return detector


def parse_device_arg(value: str) -> Device:
    try:
        return Device(value)
    except RuntimeError:
        raise ArgumentTypeError(f"'{value}' is not a valid device name.")


if __name__ == "__main__":
    # Example usage: python watermarking.py --device cuda:0 detect [file.wav]

    parser = ArgumentParser(description="Handle the watermarking for audios")
    parser.add_argument(
        "--device",
        default="cpu",
        type=parse_device_arg,
        help="device on which to run tests (default: %(default)s)",
    )
    parser.add_argument(
        "--model-file",
        default="seamlesswatermark.yaml",
        type=str,
        help="path to a config or checkpoint file (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        help="inline argument to override the value `checkpoint` specified in the file `model-file`",
    )
    sub_parser = parser.add_subparsers(title="actions", dest="sub_cmd")
    detect_parser = sub_parser.add_parser("detect")
    wm_parser = sub_parser.add_parser("wm")
    parser.add_argument("file", type=str, help="Path to the .wav file")

    args = parser.parse_args()

    if args.sub_cmd == "detect":
        model = model_from_checkpoint(args.model_file, checkpoint=args.checkpoint, device=args.device)
        wav, _ = torchaudio.load(args.file)
        wav = wav.unsqueeze(0)
        wav = wav.to(args.device)
        detection = model.detect_watermark(wav)
        print(detection[:, 1, :])
        print(torch.count_nonzero(torch.gt(detection[:, 1, :], 0.5)))
