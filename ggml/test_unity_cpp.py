import ggml
import ctypes
import torch
import pytest
import numpy as np
import torch
import fairseq2.nn
import fairseq2.nn.transformer
import logging
import sys
import functools
from pathlib import Path
from ctypes_utils import Ptr
from ctypes import c_void_p
from typing import Any
from pathlib import Path
from typing import Iterator
from ggml import NativeObj
from ggml_convert import convert_model
from seamless_communication.models.inference.translator import Translator, Modality
from fairseq2.data.audio import WaveformToFbankConverter
import torchaudio
from fairseq2.models.wav2vec2.feature_extractor import Wav2Vec2FbankFeatureExtractor
Ctx = ggml.ggml_context_p

UNITY_MODELS = Path(__file__).parent / "examples/unity/models"
CTX_PARAMS = ggml.ggml_init_params(mem_size=1024 * 1024 * 1024, mem_buffer=None)

FAIRSEQ2_CPP = Path(__file__).parent / "examples/unity/fairseq2.cpp"
UNITY_FLASH_ATTN = "\n# define UNITY_FLASH_ATTN 0\n" not in FAIRSEQ2_CPP.read_text()


@pytest.fixture(name="ctx")
def _ctx() -> Iterator[Ctx]:
    """Allocate a new context with 1024 MB of memory"""
    try:
        ctx = ggml.ggml_init(params=CTX_PARAMS)
        with torch.inference_mode():
            yield ctx
    finally:
        ggml.ggml_free(ctx)


@functools.lru_cache()
def _load_g_model_once() -> NativeObj:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"
    if not model_file.exists():
        convert_model("seamlessM4T_medium", model_file)
    return ggml.load_unity_ggml_file(model_file)

@pytest.fixture()
def g_model(ctx: Ctx) -> c_void_p:
    model = _load_g_model_once()
    ggml.lib.fairseq2_model_set_inference_ctx(model.ptr, ctx)
    return model.ptr


@functools.lru_cache(maxsize=1)
def load_translator() -> Translator:
    return Translator(
        "seamlessM4T_medium", "vocoder_36langs", torch.device("cpu"), torch.float32
    )

def load_pt_model() -> Any:
    return load_translator().model


@pytest.mark.xfail(reason="TODO")
def test_hparams_code_is_up_to_date() -> None:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"

    hparams_header_file = model_file.with_suffix(".hparams.h")
    hparams_struct = hparams_header_file.read_text().strip()
    actual_code = (UNITY_MODELS.parent / "unity_model_loader.h").read_text()
    assert hparams_struct in actual_code


def test_causal_attention_mask(ctx: Ctx):
    x = torch.zeros((1, 10, 32))
    generator = fairseq2.nn.transformer.CausalAttentionMaskGenerator()
    mask_exp = generator(x).numpy()

    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    mask = ggml.to_numpy(gmask)

    gf = ggml.ggml_build_forward(gmask)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    assert mask_exp.shape == (10, 10)
    assert mask.shape == (10, 10)
    assert np.all(mask == mask_exp)

    x = x[:, :8, :]
    mask_exp = generator(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    mask = ggml.to_numpy(gmask)

    gf = ggml.ggml_build_forward(gmask)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    assert mask_exp.shape == (8, 8)
    assert mask.shape == (8, 8)
    assert np.all(mask == mask_exp)


def test_LayerNorm_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    y_exp = pt_model.text_encoder.layers[0].ffn_layer_norm(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("LayerNorm", g_model, "text_encoder.layers.0.ffn_layer_norm", gx)
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def test_Linear_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    y_exp = pt_model.text_encoder.layers[0].ffn.inner_proj(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("Linear", g_model, "text_encoder.layers.0.ffn.inner_proj", gx)
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def test_FeedForwardNetwork_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))  # (bs, seq_len, model_dim)
    torch.nn.init.uniform_(x, -1 / 32, 1 / 32)

    # Test FFN without LayerNorm
    pt_model = load_pt_model()
    y_exp = pt_model.text_encoder.layers[0].ffn(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward(
        "StandardFeedForwardNetwork", g_model, "text_encoder.layers.0.ffn", gx
    )
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def _name(tensor: ggml.ggml_tensor_p) -> bytes:
    try:
        return tensor.contents.name  # type: ignore[no-any-return]
    except ValueError:
        return b"???"


def test_MultiheadAttention_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    self_attn = pt_model.text_encoder.layers[0].self_attn

    # Note: we use different lengths for queries and keys,
    # this tests the implementation in decoding context too.
    # Note2: ggml_flash_attn requires that we have more keys than queries
    gxq = ggml.from_numpy(ctx, x[:, :11, :].contiguous())
    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "MultiheadAttention",
        g_model,
        "text_encoder.layers.0.self_attn",
        gxq,
        gx,
        gx,
        None,  # TODO: tests with causal attention masks
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    q_exp = self_attn.q_proj(x[:, :11, :]).numpy()

    y = ggml.to_numpy(gy)
    nodes = {}

    for i in range(gf.n_nodes):
        name = _name(gf.nodes[i])
        children = [_name(gf.nodes[i].contents.src[j]) for j in range(2)]
        print(name, f"op({gf.nodes[i].contents.op})", children)
        nodes[name] = ggml.to_numpy(gf.nodes[i])

    attn_weights_hook = fairseq2.nn.transformer.StoreAttentionWeights([])
    self_attn.register_attn_weight_hook(attn_weights_hook)

    y_exp = self_attn(x[:, :11, :], None, x, x).numpy()

    q = nodes[b"q"]
    assert q.shape == q_exp.shape
    assert np.allclose(q_exp, q, atol=1e-5)

    # with flash_attn we don't have attn_weights
    if not UNITY_FLASH_ATTN:
        attn_weights = nodes[b"attn_weights"]
        [attn_weights_exp] = attn_weights_hook._storage
        attn_weights_exp = attn_weights_exp.numpy()
        assert attn_weights_exp.shape == attn_weights.shape
        # GGML is very agressively reducing small softmax weights to 0,
        # so the error isn't that small
        assert np.allclose(attn_weights_exp, attn_weights, atol=1e-3)
        # But the sums should be close to 1
        assert np.allclose(np.sum(attn_weights, axis=-1), np.ones((2 * 16, 11)))
        # And the maximum index should match the original ones.
        assert np.allclose(
            np.argmax(attn_weights_exp, axis=-1), np.argmax(attn_weights, axis=-1)
        )
    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)


def test_StandardTransformerEncoderLayer_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    x = torch.empty((2, 21, 1024))
    padding_mask = torch.ones((2, 21))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    layer = pt_model.text_encoder.layers[0]

    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask)
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardTransformerEncoderLayer",
        g_model,
        "text_encoder.layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, padding_mask)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)

def test_StandardConformerEncoderLayer_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    x = torch.load("/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/dev/seqs_before_conformer_block.pt")
    padding_mask = torch.ones((1, x.shape[1]))
    layer = pt_model.speech_encoder.inner.layers[0]
    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask[0])
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardConformerEncoderLayer",
        g_model,
        "speech_encoder.inner.layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, padding_mask)
    y_exp = y_exp.numpy()  
    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=2e-3)

def test_StandardConformerEncoderAdaptorLayer_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    x = torch.load("/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/dev/seqs_before_adaptor.pt")
    layer = pt_model.speech_encoder.adaptor_layers[0]
    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoderAdaptorLayer",
        g_model,
        "speech_encoder.adaptor_layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, None)
    y_exp = y_exp.numpy() 

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=2e-3)


def test_StandardTransformerEncoder_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    x = torch.empty((2, 21, 1024))
    padding_mask = torch.ones((2, 21))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask)
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardTransformerEncoder",
        g_model,
        "text_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_encoder(x, padding_mask)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4)

def test_StandardConformerEncoder_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    wav, _ = torchaudio.load("/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/test.wav")
    gx = ggml.from_numpy(ctx, wav * 2**15) # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoder",
        g_model,
        "speech_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
    )
    converter_input = {
        "waveform": wav.transpose(0, 1),
        "sample_rate": 16000.,
        "format": -1,
    }

    y = ggml.to_numpy(gy)
    speech_encoder_input = pt_model.speech_encoder_frontend(converter(converter_input)["fbank"].unsqueeze(0), None)[0]

    y_exp, _ = pt_model.speech_encoder(speech_encoder_input, None)
    y_exp = y_exp.numpy()  # remove batch dimension

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-2) # There are 10 elements in a 137*1024 tensor with error >1e-2

    

def test_WaveformToFbank_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
    )
    extractor = Wav2Vec2FbankFeatureExtractor(80, 2, 1)
    wav, _ = torchaudio.load("/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/test.wav")
    gx = ggml.from_numpy(ctx, wav * 2**15) # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")
    
    gy = ggml.forward(
        "WaveformToFbank",
        g_model,
        "",
        gx
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)
    converter_input = {
        "waveform": wav.transpose(0, 1),
        "sample_rate": 16000.,
        "format": -1,
    }
    y_exp = extractor(converter(converter_input)["fbank"].unsqueeze(0), None)[0]
    y_exp = y_exp.numpy() 

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=4e-3) # reduce? error is from standardization


def test_causal_attention_mask(ctx: Ctx):
    x = torch.zeros((5, 10))
    generator = fairseq2.nn.transformer.CausalAttentionMaskGenerator()
    mask_exp = generator(x)

    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    mask = ggml.to_numpy(gmask)

    assert mask_exp.shape == (10, 10)
    assert mask.shape == (10, 10)
    assert np.allclose(mask, mask_exp)


def test_PositionalEmbedding_forward(ctx: Ctx, g_model: c_void_p) -> None:
    seq = torch.zeros((4, 20, 1024), dtype=torch.float32)
    # this _legacy_pad_idx is suspicious. Shouldn't the model use 1 ? But
    # this is consistent with pt_model.text_decoder_frontend.pos_encoder._sin_offset
    pos_encoder = fairseq2.nn.SinusoidalPositionEncoder(1024, 55, _legacy_pad_idx=0)
    y_exp = pos_encoder(seq, None)[0].numpy()

    gseq = ggml.from_numpy(ctx, seq[0].numpy())
    ggml.ggml_set_name(gseq, b"seq")
    gy = ggml.forward(
        "PositionalEmbedding", g_model, "text_decoder_frontend.pos_encoder", gseq
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    y = ggml.to_numpy(gy)

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-6)


def test_TransformerEmbeddingFrontend_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    seq = torch.arange(2 * 20).reshape(2, 20)
    seq[1, 15:] = 0  # padding for second sentence
    seq_len = torch.tensor([20, 15])
    gseq = ggml.from_numpy(ctx, seq.numpy().astype(np.int32))

    ggml.ggml_set_name(gseq, b"seq")
    gy = ggml.forward(
        "TransformerEmbeddingFrontend", g_model, "text_decoder_frontend", gseq
    )
    ggml.build_and_compute(ctx, gy)
    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_decoder_frontend(seq, seq_len)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-6)


def test_StandardTransformerDecoder_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    x = torch.empty((2, 13, 1024))
    encoder_out = torch.empty((2, 21, 1024))
    padding_mask = torch.ones((2, 13))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)
    torch.nn.init.uniform_(encoder_out, -1, 1)
    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask)
    ggml.ggml_set_name(gpad, b"padding_mask")
    genc = ggml.from_numpy(ctx, encoder_out)
    gy = ggml.forward(
        "StandardTransformerDecoder",
        g_model,
        "text_decoder",
        gx,
        None,  # TODO support padding mask,
        genc,
        None,
    )
    ggml.build_and_compute(ctx, gy)
    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_decoder(x, padding_mask, encoder_out, None)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-3)


def test_t2tt(ctx: Ctx, g_model: c_void_p):
    src_lang = "eng"
    src_text = "We are all in a yellow submarine."
    tgt_lang = "fra"
    sample_file = Path(__file__).parent / "sample_input.npz"
    beam_size = 2

    if not sample_file.exists():
        translator = load_translator()
        device = translator.device
        token_encoder = translator.text_tokenizer.create_encoder(
            task="translation", lang=src_lang, mode="source", device=device
        )
        src = translator.collate(token_encoder(src_text))

        text_out, _ = translator.get_prediction(
            translator.model,
            translator.text_tokenizer,
            translator.unit_tokenizer,
            src,
            input_modality=Modality.TEXT,
            output_modality=Modality.TEXT,
            tgt_lang=tgt_lang,
            beam_size=beam_size,
        )

        tgt_text = str(text_out.sentences[0])
        assert tgt_text == "Nous sommes tous dans un sous-marin jaune."
        hypotheses = [
            {
                "seq": h.seq.tolist(),
                "score": h.score.item(),
                "step_scores": h.step_scores.numpy(),
            }
            for h in text_out.generator_output.results[0]
        ]
        np.savez(
            sample_file,
            encoder_output=text_out.encoder_output.numpy(),
            encoder_padding_mask=text_out.encoder_padding_mask.numpy(),
            hypotheses=hypotheses,
        )

    # allow_pickle to load the hyp dicts
    text_out = np.load(sample_file, allow_pickle=True)
    encoder_out = ggml.from_numpy(ctx, text_out["encoder_output"])
    encoder_padding_mask = ggml.from_numpy(ctx, text_out["encoder_padding_mask"])
    prefix_seq = np.array(text_out["hypotheses"][0]["seq"][:2]).astype(np.int32)
    max_seq_len = max(len(h["seq"]) for h in text_out["hypotheses"])

    job = ggml.SequenceGeneratorJob()
    job.opts.beam_size = beam_size
    job.opts.min_seq_len = 1
    job.opts.soft_max_seq_len_a = 1
    job.opts.soft_max_seq_len_b = 200
    job.opts.hard_max_seq_len = int(max_seq_len * 1.5)
    job.opts.len_penalty = 1.0
    job.opts.unk_penalty = 0.0
    job.opts.normalize_scores = True

    job.prefix_seq = ggml.from_numpy(ctx, prefix_seq)
    job.pad_idx = 0
    job.unk_idx = 1
    job.bos_idx = 2
    job.eos_idx = 3

    result_ptr = ggml.generate_sequence(
        g_model, job, encoder_out, encoder_padding_mask, ctx
    )
    results = [result_ptr[i] for i in range(beam_size) if result_ptr[i].seq != None]

    assert len(results) == len(text_out["hypotheses"])
    for g_hyp, exp in zip(results, text_out["hypotheses"]):
        g_tokens = list(ggml.to_numpy(g_hyp.seq))
        g_step_scores = ggml.to_numpy(g_hyp.step_scores)
        assert g_tokens == exp["seq"]
        assert g_hyp.score == pytest.approx(exp["score"], rel=1e-2)
        # The score error is big, this may negatively impact the beam search.
        assert np.allclose(g_step_scores, exp["step_scores"], atol=0.1)

def test_s2tt(ctx: Ctx, g_model: c_void_p):
    src_audio_wav, _ = torchaudio.load("/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/test.wav")
    # translator = load_translator()
    # token_encoder = translator.text_tokenizer.create_encoder(
    #     task="translation"
    # )
    # decoded_audio = {
    #     "waveform": src_audio_wav.t(),
    #     "sample_rate": 16000.,
    #     "format": -1,
    # }
    # src = translator.collate(translator.convert_to_fbank(decoded_audio))["fbank"]

    # text_out, _ = translator.get_prediction(
    #     translator.model,
    #     translator.text_tokenizer,
    #     translator.unit_tokenizer,
    #     src,
    #     input_modality=Modality.SPEECH,
    #     output_modality=Modality.TEXT,
    #     tgt_lang="cmn",
    # )

    # tgt_text = str(text_out.sentences[0])
    # assert tgt_text == "大家好 , 世界无主题。"
    # tgt_tokens = text_out.generator_output.results[0][0].seq
    # score = text_out.generator_output.results[0][0].score.item()

    tgt_tokens = [     3, 256200,  16991, 249346, 249725,    146,  25220, 251069, 249211,
        251148, 253935,      3]
    score = -1.606838583946228
    gx = ggml.from_numpy(ctx, src_audio_wav * 2**15) # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoder",
        g_model,
        "speech_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    encoder_out = gy

    job = ggml.SequenceGeneratorJob()
    job.opts.beam_size = 1
    job.opts.min_seq_len = 1
    job.opts.soft_max_seq_len_a = 1
    job.opts.soft_max_seq_len_b = 200
    job.opts.hard_max_seq_len = 20
    job.opts.len_penalty = 1.0
    job.opts.unk_penalty = 0.0
    job.prefix_seq = ggml.from_numpy(ctx, np.array([3, 256200]).astype(np.int32))
    job.opts.normalize_scores = True
    job.pad_idx = 0
    job.unk_idx = 1
    job.bos_idx = 2
    job.eos_idx = 3

    result = ggml.ggml_tensor()
    
    g_score = ggml.generate_sequence(
        g_model, job, encoder_out, None, ctypes.byref(result)
    )
    tokens = list(ggml.to_numpy(result))
    assert tokens == tgt_tokens
    assert g_score == pytest.approx(score)
