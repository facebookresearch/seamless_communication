import torch
from datasets import load_dataset
from jiwer import wer
import os
from typing import Tuple, Iterable, Dict, Any
import logging
from whisper.normalizers import EnglishTextNormalizer

logging.basicConfig(level=logging.INFO)

from seamless_communication.models.unity import UnitYModel
from seamless_communication.inference import Translator

log = logging.getLogger("l")

TOKEN = "dummy"
MAX_SAMPLES = 100
CHCK_PATH = os.path.expanduser("~/tune_chck/chck.pt")

norm = EnglishTextNormalizer()


DATASET = []  # type:ignore


def __iterate_test_ds() -> Iterable[Tuple[torch.Tensor, str]]:
    ds = load_dataset(
        "speechcolab/gigaspeech",
        "s",
        token=os.environ.get("HF_TOKEN", TOKEN),
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    for idx, item in enumerate(ds):
        if idx >= MAX_SAMPLES:
            break
        assert item["audio"]["sampling_rate"] == 16000
        yield (torch.from_numpy(item["audio"]["array"]), item["text"])


def _iterate_test_ds() -> Iterable[Tuple[torch.Tensor, str]]:
    global DATASET
    if not DATASET:
        DATASET = list(__iterate_test_ds())
    yield from DATASET


def _eval(translator: Translator) -> float:
    references = []
    predictions = []
    for idx, (wav, text) in enumerate(_iterate_test_ds()):
        reference = norm(text)
        if not reference:
            reference = "."
        references.append(reference)
        prediction = str(
            translator.predict(
                input=wav,
                task_str="s2tt",
                tgt_lang="eng",
                src_lang="eng",
            )[0][0]
        )
        prediction = norm(prediction)
        if not prediction:
            prediction = "."
        log.info(idx)
        log.info(f"REF: {reference}")
        log.info(f"PRE: {prediction}")
        log.info("----")
        predictions.append(prediction)
    return wer(reference=references, hypothesis=predictions)  # type:ignore


def _select_keys(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}


def load_checkpoint(model: UnitYModel, chck_path: str) -> None:
    state_dict = torch.load(chck_path, map_location="cpu")
    model.speech_encoder_frontend.load_state_dict(_select_keys(state_dict, "model.speech_encoder_frontend."))
    model.speech_encoder.load_state_dict(_select_keys(state_dict, "model.speech_encoder."))
    assert model.text_decoder_frontend is not None
    model.text_decoder_frontend.load_state_dict(_select_keys(state_dict, "model.text_decoder_frontend."))
    assert model.text_decoder is not None
    model.text_decoder.load_state_dict(_select_keys(state_dict, "model.text_decoder."))


def main() -> None:
    translator = Translator(
        model_name_or_card="seamlessM4T_medium",
        vocoder_name_or_card=None,
        device=torch.device("cuda"),
    )
    non_tuned_wer = _eval(translator)

    load_checkpoint(translator.model, CHCK_PATH)
    tuned_wer = _eval(translator)

    log.info(f"WER non-tuned: {non_tuned_wer:.3f}")
    log.info(f"WER tuned: {tuned_wer:.3f}")


if __name__ == "__main__":
    main()
