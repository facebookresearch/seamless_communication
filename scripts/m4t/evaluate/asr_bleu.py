# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os

import sacrebleu
import torch
import torchaudio
import whisper
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from seamless_communication.models.inference import Translator
from seamless_communication.models.inference.translator import Modality
from seamless_communication.models.unity import (UnitYModel, load_unity_model,
                                                 load_unity_text_tokenizer,
                                                 load_unity_unit_tokenizer)
from seamless_communication.models.vocoder import Vocoder, load_vocoder_model
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer


class ASRBleu:
    def __init__(
        self,
        output_path: str,
    ):
        self.output_path = output_path
        waveforms_dir = os.path.join(output_path, "output_waveforms")
        os.makedirs(waveforms_dir, exist_ok=True)

    def compute_asr_bleu(
        self,
        input_path: str,
        reference_path: str,
        tgt_lang: str,
        src_lang: str,
        audio_format: str,
        dataset_name: str,
        save_first_pass: bool,
        model_name: str,
        device: str,
        dtype: str,
    ):
        # Retrieve ground truth reference text
        reference = []
        normalizer = (
            EnglishTextNormalizer() if tgt_lang == "eng" else BasicTextNormalizer()
        )
        with open(reference_path, "r") as reference_file:
            for line in reference_file.readlines():
                reference.append(normalizer(line))

        # Initialize the multitask model
        model: UnitYModel = Translator.load_model_for_inference(
            load_model_fn=load_unity_model,
            model_name_or_card=model_name,
            device=device,
            dtype=dtype,
        )

        text_tokenizer = load_unity_text_tokenizer(model_name)
        unit_tokenizer = load_unity_unit_tokenizer(model_name)
        decode = AudioDecoder(torch.float32, device)
        collate = Collater(pad_idx=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2)
        convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=device,
            dtype=dtype,
        )

        # Generate and save text and unit outputs
        text_out = []
        unit_out = []
        unit_file_name = (
            self.output_path + f"/generate-{dataset_name}_{src_lang}-{tgt_lang}.unit"
        )
        first_pass_file_name = (
            self.output_path
            + f"/first-pass-{dataset_name}_{src_lang}-{tgt_lang}_ref_pred.txt"
        )
        with open(unit_file_name, "w+") as unit_file:
            if save_first_pass:
                first_pass_file = open(first_pass_file_name, "w+")
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
                    if save_first_pass:
                        first_pass_file.write(f"{text_out[i]}\n")
                    unit_out.append(result[1])
                    unit_file.write(f"{unit_out[i]}\n")
                except FileNotFoundError:
                    break

        # First pass BLEU score computation and save
        tokenizer = "char" if tgt_lang in ["cmn", "jpn", "tha", "lao", "mya"] else "13a"
        bleu_metric = sacrebleu.BLEU(tokenize=tokenizer)
        if save_first_pass:
            first_pass_file.close()
            bleu_score = bleu_metric.corpus_score(text_out, [reference])
            first_pass_bleu = (
                self.output_path
                + f"/{dataset_name}_{src_lang}-{tgt_lang}_first_pass_bleu.json"
            )
            with open(first_pass_bleu, "w+") as f:
                f.write(
                    bleu_score.format(
                        signature=str(bleu_metric.get_signature()), is_json=True
                    )
                )

        # Free GPU memory
        torch.cuda.empty_cache()

        # Initialize the vocoder
        vocoder: Vocoder = Translator.load_model_for_inference(
            load_model_fn=load_vocoder_model,
            model_name_or_card="vocoder_36langs",
            device=device,
            dtype=dtype,
        )

        # Generate and save audio
        for i, unit in enumerate(unit_out):
            units = unit.units[:, 1:][0].cpu().numpy().tolist()
            wav_out = vocoder(units, tgt_lang, -1, dur_prediction=True)
            wav_file_name = self.output_path + f"/output_waveforms/{i}_pred.wav"
            with open(wav_file_name, "w+") as _:
                pass
            torchaudio.save(
                wav_file_name,
                wav_out[0].to(torch.float32).cpu(),
                sample_rate=16000,
            )

        # Free GPU memory
        torch.cuda.empty_cache()

        # Initialize the Whisper model
        whisper_model = whisper.load_model("base")

        # Generate and save transcriptions
        transcriptions = []
        ref_pred_file_name = (
            self.output_path + f"/{dataset_name}_{src_lang}-{tgt_lang}_ref_pred.tsv"
        )
        with open(ref_pred_file_name, "w+") as transcriptions_file:
            for i, reference_line in enumerate(reference):
                filename = "n_pred.wav".replace("n", str(i))
                transcription = normalizer(
                    whisper_model.transcribe(
                        self.output_path + f"/output_waveforms/{filename}",
                        temperature=0,
                        beam_size=1,
                    )["text"]
                )
                transcriptions.append(transcription)
                transcriptions_file.write(f"{i}\t{reference_line}\t{transcription}\n")

        # Compute and save BLEU score
        bleu_score = bleu_metric.corpus_score(transcriptions, [reference])
        bleu_filename = (
            self.output_path + f"/{dataset_name}_{src_lang}-{tgt_lang}_bleu.json"
        )
        with open(bleu_filename, "w+") as f:
            f.write(
                bleu_score.format(
                    signature=str(bleu_metric.get_signature()), is_json=True
                )
            )
