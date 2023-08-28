# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sacrebleu
import torch
import torchaudio
import whisper
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock

from seamless_communication.models.inference import Translator
from seamless_communication.models.inference.translator import Modality
from seamless_communication.models.vocoder import load_vocoder_model, Vocoder
from seamless_communication.models.unity import (
    UnitYModel,
    UnitYGenerator,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

class ASRBleu:
    def __init__(
        self,
        output_path: str,
    ):
        self.output_path = output_path
        waveforms_dir = os.path.join(output_path, 'output_waveforms')
        os.makedirs(waveforms_dir, exist_ok=True)

    def compute_asr_bleu(
        self,
        input: str,
        tgt_lang: str,
        src_lang: str,
        model_name: str,
        device: str,
        dtype: str                
    ):
        # Initialize the multitask model
        model: UnitYModel = Translator.load_model_for_inference(
            load_model_fn=load_unity_model,
            model_name_or_card=model_name,
            device=device,
            dtype=dtype,
        )

        text_tokenizer = load_unity_text_tokenizer(model_name)
        unit_tokenizer = load_unity_unit_tokenizer(model_name)

        with open(input, "rb") as input_file:
            block = MemoryBlock(input_file.read())
        decode = AudioDecoder(torch.float32, device)
        collate = Collater(
            pad_idx=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )
        convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=device,
            dtype=dtype,
        )
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
        text_out, unit_out = str(result[0].sentences[0]), result[1]
        normalizer = EnglishTextNormalizer() if tgt_lang == "eng" else BasicTextNormalizer()
        text_out = normalizer(text_out)

        # Run a torch.cuda.empty_cache() to free GPU memory
        torch.cuda.empty_cache()

		# Initialize the vocoder only after multitask inference is completed.
        self.vocoder: Vocoder = Translator.load_model_for_inference(
            load_model_fn=load_vocoder_model,
            model_name_or_card="vocoder_36langs",
            device=device,
            dtype=dtype,
        )
        units = unit_out.units[:, 1:][0].cpu().numpy().tolist()
        wav_out = self.vocoder(units, tgt_lang, -1, dur_prediction=True)

        # Save audio
        with open(self.output_path + "/output_waveforms/audio.wav", "w") as _:
            pass
        torchaudio.save(
            self.output_path + "/output_waveforms/audio.wav",
            wav_out[0].to(torch.float32).cpu(),
            sample_rate=16000,
        )

		# Run a torch.cuda.empty_cache() to free GPU memory	
        torch.cuda.empty_cache()

		# Initialize the Whisper model only after vocoder inference is completed.
        whisper_model = whisper.load_model("base")
        transcription = whisper_model.transcribe(self.output_path + "/output_waveforms/audio.wav", temperature=0, beam_size=1)["text"]

        # Compute BLEU score
        #TODO: Figure out char level encoding for below langs
        tokenizer = None if tgt_lang in ["cmn", "jpn", "tha", "lao", "mya"] else '13a'
        bleu_metric = sacrebleu.BLEU(tokenize=tokenizer)
        bleu_score = bleu_metric.corpus_score([transcription], [[text_out]])
        # bleu_score = sacrebleu.corpus_bleu([transcription], [[text_out]])
        
        # Write results to output directory
        with open(self.output_path + f"generate-{src_lang}-{tgt_lang}.unit", "w") as transcriptions_file:
            transcriptions_file.write(transcription)
        with open(self.output_path + f"{src_lang}-{tgt_lang}_ref_pred.tsv", "w") as references_file:
            references_file.write(text_out)
        with open(self.output_path + f"{src_lang}-{tgt_lang}_bleu.json", "w") as bleu_score_file:
            bleu_score_dict = {
                "score": bleu_score.score,
                "counts": bleu_score.counts,
                "totals": bleu_score.totals,
                "precisions": bleu_score.precisions,
                "bp": bleu_score.bp,
                "sys_len": bleu_score.sys_len,
                "ref_len": bleu_score.ref_len,
            }
            json.dump(bleu_score_dict, bleu_score_file)
