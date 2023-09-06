# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os

import sacrebleu
import torch
import torchaudio
import whisper
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from m4t_scripts.evaluate.download_data import download_datasets
from seamless_communication.models.inference import Translator
from seamless_communication.models.inference.translator import Modality
from seamless_communication.models.unity import (UnitYModel, load_unity_model,
                                                 load_unity_text_tokenizer,
                                                 load_unity_unit_tokenizer)
from seamless_communication.models.vocoder import Vocoder, load_vocoder_model
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class ASRBleu:
    def __init__(
        self,
        output_dir: str,
    ):
        # Set up output directory
        self.output_dir = output_dir
        waveforms_dir = os.path.join(output_dir, "output_waveforms")
        os.makedirs(waveforms_dir, exist_ok=True)

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.dtype = torch.float16
            logger.info(f"Running inference on the GPU in {self.dtype}.")
        else:
            self.device = torch.self.device("cpu")
            self.dtype = torch.float32
            logger.info(f"Running inference on the CPU in {self.dtype}.")

    def _compute_inference(
        self,
        model_name: str,
        tgt_lang: str,
        eval_first_pass: bool,
        audio_format: str,
        input_path: str,
        units_path: str,
        first_pass_path: str,
    ):
        """SeamlessM4T model runs inference on selected dataset"""

        # Initialize the multitask model
        model: UnitYModel = Translator.load_model_for_inference(
            load_model_fn=load_unity_model,
            model_name_or_card=model_name,
            device=self.device,
            dtype=self.dtype,
        )

        # Misc functionality required for inference
        text_tokenizer = load_unity_text_tokenizer(model_name)
        unit_tokenizer = load_unity_unit_tokenizer(model_name)
        decode = AudioDecoder(torch.float32, self.device)
        collate = Collater(pad_idx=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2)
        convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=self.device,
            dtype=self.dtype,
        )

        # Generate and save text and unit outputs
        text_out = []
        unit_out = []
        with open(units_path, "w+") as unit_file:
            if eval_first_pass:
                first_pass_file = open(first_pass_path, "w+")
            for i in itertools.count():
                name = audio_format.replace("n", str(i))
                try:
                    with open(input_path + f"/{name}", "rb") as input_file:
                        block = MemoryBlock(input_file.read())
                    src = collate(convert_to_fbank(decode(block)))["fbank"]
                    result = Translator.get_prediction(
                        model=model,
                        text_tokenizer=text_tokenizer,
                        unit_tokenizer=unit_tokenizer,
                        src=src,
                        input_modality=Modality.SPEECH,
                        output_modality=Modality.SPEECH,
                        tgt_lang=tgt_lang,
                    )
                    text_out.append(str(result[0].sentences[0]))
                    if eval_first_pass:
                        first_pass_file.write(f"{text_out[i]}\n")
                    unit_out.append(result[1])
                    unit_file.write(f"{unit_out[i]}\n")
                except FileNotFoundError:
                    break

        if eval_first_pass:
            first_pass_file.close()

        # Free GPU memory
        torch.cuda.empty_cache()

        return unit_out, text_out

    def _generate_audio(
        self,
        unit_out,
        tgt_lang,
        wav_path,
    ):
        """Vocoder generates audio based on generated units"""

        # Initialize the vocoder
        vocoder: Vocoder = Translator.load_model_for_inference(
            load_model_fn=load_vocoder_model,
            model_name_or_card="vocoder_36langs",
            device=self.device,
            dtype=self.dtype,
        )

        # Generate and save audio
        for i, unit in enumerate(unit_out):
            units = unit.units[:, 1:][0].cpu().numpy().tolist()
            wav_out = vocoder(units, tgt_lang, -1, dur_prediction=True)
            wav_file_path = wav_path + f"{i}_pred.wav"
            with open(wav_file_path, "w+") as _:
                pass
            torchaudio.save(
                wav_file_path,
                wav_out[0].to(torch.float32).cpu(),
                sample_rate=16000,
            )

        # Free GPU memory
        torch.cuda.empty_cache()

    def _compute_asr(
        self,
        normalizer,
        reference,
        wav_path,
        ref_pred_path,
    ):
        """Compute ASR on generated audio"""

        # Initialize the Whisper model
        whisper_model = whisper.load_model("base")

        # Generate and save transcriptions
        transcriptions = []
        with open(ref_pred_path, "w+") as transcriptions_file:
            for i, reference_line in enumerate(reference):
                transcription = normalizer(
                    whisper_model.transcribe(
                        wav_path + f"{i}_pred.wav",
                        temperature=0,
                        beam_size=1,
                    )["text"]
                )
                transcriptions.append(transcription)
                transcriptions_file.write(f"{i}\t{reference_line}\t{transcription}\n")

        return transcriptions

    def _compute_bleu(
        self,
        reference,
        text_out,
        transcriptions,
        tgt_lang,
        eval_first_pass,
        first_pass_bleu,
        bleu_file_path,
    ):
        """Compute and save BLEU scores"""
        tokenizer = "char" if tgt_lang in ["cmn", "jpn", "tha", "lao", "mya"] else "13a"
        bleu_metric = sacrebleu.BLEU(tokenize=tokenizer)
        if eval_first_pass:
            bleu_score = bleu_metric.corpus_score(text_out, [reference])
            with open(first_pass_bleu, "w+") as f:
                f.write(
                    bleu_score.format(
                        signature=str(bleu_metric.get_signature()), is_json=True
                    )
                )
        bleu_score = bleu_metric.corpus_score(transcriptions, [reference])
        with open(bleu_file_path, "w+") as f:
            f.write(
                bleu_score.format(
                    signature=str(bleu_metric.get_signature()), is_json=True
                )
            )

    def compute_asr_bleu(
        self,
        lang_dir: str,
        split: str,
        num_data_pairs: int,
        model_name: str,
        eval_first_pass: bool,
        dataset: str,
        audio_format: str,
    ):
        """BLEU score evaluation for SeamlessM4T models on S2ST tasks"""

        # Download fleurs test data
        src_lang, tgt_lang = lang_dir.split("-")
        download_datasets([(src_lang, tgt_lang)], split, num_data_pairs, "./data")

        # Input paths
        input_path = f"./data/{lang_dir}/source_audio_{src_lang}/"
        reference_path = f"./data/{lang_dir}/target_texts_{tgt_lang}/references.txt"

        # Output paths
        units_path = self.output_dir + f"/generate-{dataset}_{src_lang}-{tgt_lang}.unit"
        first_pass_path = (
            self.output_dir
            + f"/first-pass-{dataset}_{src_lang}-{tgt_lang}_ref_pred.txt"
        )
        first_pass_bleu = (
            self.output_dir + f"/{dataset}_{src_lang}-{tgt_lang}_first_pass_bleu.json"
        )
        ref_pred_path = (
            self.output_dir + f"/{dataset}_{src_lang}-{tgt_lang}_ref_pred.tsv"
        )
        wav_path = self.output_dir + "/output_waveforms/"
        bleu_file_path = self.output_dir + f"/{dataset}_{src_lang}-{tgt_lang}_bleu.json"

        # Retrieve ground truth reference text
        reference = []
        normalizer = (
            EnglishTextNormalizer() if tgt_lang == "eng" else BasicTextNormalizer()
        )
        with open(reference_path, "r") as reference_file:
            for line in reference_file.readlines():
                reference.append(normalizer(line))

        # Run inference
        unit_out, text_out = self._compute_inference(
            model_name,
            tgt_lang,
            eval_first_pass,
            audio_format,
            input_path,
            units_path,
            first_pass_path,
        )

        # Generate audio based on the units
        self._generate_audio(
            unit_out,
            tgt_lang,
            wav_path,
        )

        # Run ASR on generated audio
        transcriptions = self._compute_asr(
            normalizer,
            reference,
            wav_path,
            ref_pred_path,
        )

        # Compute resulting BLEU scores
        self._compute_bleu(
            reference,
            text_out,
            transcriptions,
            tgt_lang,
            eval_first_pass,
            first_pass_bleu,
            bleu_file_path,
        )
