<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

![](23-11_SEAMLESS_BlogHero_11.17.jpg)
# Seamless 简介

Seamless 是一组旨在实现更自然、更真实跨语言沟通的 AI 模型家族。SeamlessM4T 是一个支持近 100 种语言的大规模多语言多模态机器翻译基础模型。SeamlessM4T 作为 SeamlessExpressive（在跨语言翻译中保留音色风格与语调韵律的模型）和 SeamlessStreaming（支持近 100 种语言的同声传译与流式语音识别 ASR 的模型）的基础。SeamlessExpressive 与 SeamlessStreaming 融为一体，构成了统一的 Seamless 模型，兼具多语言、实时流式和富有表现力的高保真跨语言语音翻译能力。

## 相关链接

### 在线体验 (Demos)

|                        | SeamlessM4T v2                                                                                                                        | SeamlessExpressive                                                                                                                               | SeamlessStreaming                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| 官方演示 Demo           | [SeamlessM4T v2 Demo](https://seamless.metademolab.com/m4t?utm_source=github&utm_medium=web&utm_campaign=seamless&utm_content=readme) | [SeamlessExpressive Demo](https://seamless.metademolab.com/expressive?utm_source=github&utm_medium=web&utm_campaign=seamless&utm_content=readme) |                                                                                          |
| HuggingFace Space 体验 | [🤗 SeamlessM4T v2 Space](https://huggingface.co/spaces/facebook/seamless-m4t-v2-large)                                                | [🤗 SeamlessExpressive Space](https://huggingface.co/spaces/facebook/seamless-expressive)                                                         | [🤗 SeamlessStreaming Space](https://huggingface.co/spaces/facebook/seamless-streaming) |

### 学术论文 (Papers)
[Seamless](https://ai.facebook.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)

[EMMA](https://ai.meta.com/research/publications/efficient-monotonic-multihead-attention/)

[SONAR](https://ai.meta.com/research/publications/sonar-expressive-zero-shot-expressive-speech-to-speech-translation/)

### 官方博客
[AI at Meta Blog](https://ai.meta.com/research/seamless-communication/)

## 教程
NeurIPS 2023 - Seamless EXPO 专题研讨会上的详尽[教程 (Tutorial)](Seamless_Tutorial.ipynb)，提供了一站式上手与使用全部 Seamless 系列模型的实战指引。欢迎运行并体验该 Notebook。

## SeamlessM4T
SeamlessM4T 是我们的全能型**大**规模**多**语言**多**模态**机**器**翻**译（**M**assively **M**ultilingual and **M**ultimodal **M**achine **T**ranslation）基础模型，为近 100 种语言提供高质量的语音与文本翻译。

SeamlessM4T 支持以下任务：
- 语音到语音翻译 (S2ST)
- 语音到文本翻译 (S2TT)
- 文本到语音翻译 (T2ST)
- 文本到文本翻译 (T2TT)
- 自动语音识别 (ASR)

:star2: 我们推出了全新 SeamlessM4T v2，采用了创新的 *UnitY2* 架构。与 SeamlessM4T v1 相比，新模型在翻译质量和语音生成任务的推理延迟方面均有显著提升。

欲深入了解 SeamlessM4T 模型系列、算法机制、语言覆盖范围及基准表现，请参阅 [SeamlessM4T 说明文档](docs/m4t/README.md) 或 [🤗 Model Card](https://huggingface.co/facebook/seamless-m4t-v2-large)。

> [!NOTE]
> SeamlessM4T 现已无缝集成至 🤗 Transformers 库。详情请参阅[该章节](docs/m4t/README.md#transformers-usage)。

## SeamlessExpressive

SeamlessExpressive 是一款语音到语音翻译模型，能够捕捉此前较少被探索的韵律特征（例如语速和停顿），同时完整保留原发言者的音色风格与极高的文本翻译质量。

欲深入了解 SeamlessExpressive 模型，请参阅 [SeamlessExpressive 说明文档](docs/expressive/README.md) 或 [🤗 Model Card](https://huggingface.co/facebook/seamless-expressive)。

## SeamlessStreaming

SeamlessStreaming 是一款流式低延迟翻译模型。该模型以语音作为输入模态，支持语音与文本两种输出模态。

SeamlessStreaming 支持以下任务：
- 语音到语音翻译 (S2ST)
- 语音到文本翻译 (S2TT)
- 自动语音识别 (ASR)

欲深入了解 SeamlessStreaming 模型，请参阅 [SeamlessStreaming 说明文档](docs/streaming/README.md) 或 [🤗 Model Card](https://huggingface.co/facebook/seamless-streaming)。

## Seamless

Seamless 是集表现力韵律保留与实时流式低延迟于一身的统一端到端语音翻译模型。

## 最新更新
- [2023/12/18] 我们开源了基于 Conformer 的 [W2v-BERT 2.0 语音编码器](#w2v-bert-20-语音编码器)（参见[论文](https://arxiv.org/pdf/2312.05187.pdf)第 3.2.1 节），该编码器是 Seamless 系列模型的核心基础。
- [2023/12/14] 发布了 NeurIPS 2023 的 Seamless [官方教程](#教程)。

# 快速上手 (Quick Start)
## 环境安装
> [!NOTE]
> 前置核心依赖之一为 [fairseq2](https://github.com/facebookresearch/fairseq2)，其预编译 Wheel 仅适用于 Linux x86-64 和 Apple Silicon Mac 计算机。此外，该库依赖可能尚未安装在系统中的 [libsndfile](https://github.com/libsndfile/libsndfile)。若在安装过程中遇到问题，请查阅其 [README](https://github.com/facebookresearch/fairseq2) 获取详细指引。

```bash
pip install .
```

> [!NOTE]
> 用于计算评测指标的转录推理音频依赖于 [Whisper](https://github.com/openai/whisper#setup)（会自动安装）。Whisper 需要系统中预先安装命令行工具 [`ffmpeg`](https://ffmpeg.org/)（可通过大多数包管理器直接安装）。

## 运行推理

### SeamlessM4T 推理
以下为在项目根目录下通过 CLI 命令行运行推理的示例。

S2ST（语音到语音翻译）任务：
```bash
m4t_predict <path_to_input_audio> --task s2st --tgt_lang <tgt_lang> --output_path <path_to_save_audio>
```
T2TT（文本到文本翻译）任务：
```bash
m4t_predict <input_text> --task t2tt --tgt_lang <tgt_lang> --src_lang <src_lang>
```
详细的推理操作指南以及语音和文本模态在源语言/目标语言侧的完整支持列表，请参阅[推理说明文档](src/seamless_communication/cli/m4t/predict)。

如需使用 GGML 原生（脱离 Python 环境）高效运行 S2TT/ASR，请参阅 [unity.cpp 章节](#unitycpp)。

### SeamlessExpressive 推理
> [!NOTE]
> 关于如何下载该模型，请参阅[相关章节](#seamlessexpressive-模型)。

以下为在项目根目录下使用 CLI 运行推理的示例：

```bash
expressivity_predict <path_to_input_audio> --tgt_lang <tgt_lang> --model_name seamless_expressivity --vocoder_name vocoder_pretssel --output_path <path_to_save_audio>
```

### SeamlessStreaming 与 Seamless 推理

[流式评测说明文档](src/seamless_communication/cli/streaming)提供了针对 SeamlessStreaming 和 Seamless 模型的详尽评测指南。CLI 支持 `--no-scoring` 参数，可跳过指标评分阶段直接运行推理。

更多细节请查阅[推理文档](src/seamless_communication/inference)。

## 运行 SeamlessStreaming 在线演示
您可以复制（Duplicate）[SeamlessStreaming HF Space](https://huggingface.co/spaces/facebook/seamless-streaming?duplicate=true) 快速运行流式演示应用。

您也可以通过[此处](https://huggingface.co/spaces/facebook/seamless-streaming/tree/main)克隆 Space 仓库在本地运行演示。有关安装步骤的更多细节，请参阅 SeamlessStreaming HF 仓库的 [README](https://huggingface.co/spaces/facebook/seamless-streaming/blob/main/README.md)。

## 本地运行 SeamlessM4T 与 SeamlessExpressive [Gradio](https://github.com/gradio-app/gradio) 演示

在本地启动与 Hugging Face 上完全相同的 Demo Space：

```bash
cd demo
pip install -r requirements.txt
python app.py
```

# 资源与使用指引
## 模型权重 (Models)
### SeamlessM4T 模型
| 模型名称                 | 参数量 (#params) | 权重下载 (Checkpoint)                                                                                                                                                             | 评测指标 (Metrics)                                                                  |
| ----------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| SeamlessM4T-Large v2    | 2.3B            | [🤗 Model card](https://huggingface.co/facebook/seamless-m4t-v2-large) - [checkpoint](https://huggingface.co/facebook/seamless-m4t-v2-large/resolve/main/seamlessM4T_v2_large.pt)  | [metrics](https://dl.fbaipublicfiles.com/seamless/metrics/seamlessM4T_large_v2.zip) |
| SeamlessM4T-Large (v1)  | 2.3B            | [🤗 Model card](https://huggingface.co/facebook/seamless-m4t-large) - [checkpoint](https://huggingface.co/facebook/seamless-m4t-large/resolve/main/multitask_unity_large.pt)    | [metrics](https://dl.fbaipublicfiles.com/seamless/metrics/seamlessM4T_large.zip)    |
| SeamlessM4T-Medium (v1) | 1.2B            | [🤗 Model card](https://huggingface.co/facebook/seamless-m4t-medium) - [checkpoint](https://huggingface.co/facebook/seamless-m4t-medium/resolve/main/multitask_unity_medium.pt) | [metrics](https://dl.fbaipublicfiles.com/seamless/metrics/seamlessM4T_medium.zip)   |

### SeamlessExpressive 模型

[🤗 Model card](https://huggingface.co/facebook/seamless-expressive)

如需获取并下载 SeamlessExpressive 模型资产，请提交[此申请表单](https://ai.meta.com/resources/models-and-libraries/seamless-downloads/)。审批通过后，您将收到包含各模型权重下载链接的确认邮件。

请注意，SeamlessExpressive 的使用须遵循其专用的[开源协议](SEAMLESS_LICENSE)与[可接受使用政策 (Acceptable Use Policy)](ACCEPTABLE_USE_POLICY)。

### SeamlessStreaming 模型
| 模型名称           | 参数量 (#params) | 权重下载 (Checkpoint)                                                                                                                                                                                                                                                                                    | 评测指标 (Metrics)                                                                          |
| ----------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| SeamlessStreaming | 2.5B            | [🤗 Model card](https://huggingface.co/facebook/seamless-streaming) - [单调解码器权重](https://huggingface.co/facebook/seamless-streaming/resolve/main/seamless_streaming_monotonic_decoder.pt) - [流式 UnitY2 权重](https://huggingface.co/facebook/seamless-streaming/resolve/main/seamless_streaming_unity.pt) | [metrics](https://dl.fbaipublicfiles.com/seamless/metrics/streaming/seamless_streaming.zip) |

### Seamless 模型
Seamless 模型即为 SeamlessStreaming 模型将默认的无韵律表现力 `vocoder_v2` 替换为富有表现力的 `vocoder_pretssel` 声码器。
请查阅上述[章节](#seamlessexpressive-模型)了解如何获取 `vocoder_pretssel` 权重。

### W2v-BERT 2.0 语音编码器
| 模型名称        | 参数量 (#params) | 权重下载 (Checkpoint)                                                                                                                                                                                                                                                                                    |
| ----------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W2v-BERT 2.0 | 600M    | [🤗 Model card](https://huggingface.co/facebook/conformer-shaw) - [checkpoint](https://huggingface.co/facebook/conformer-shaw/resolve/main/conformer_shaw.pt) |

以下为调用语音编码器执行前向推理的 Python 代码示例：

```python
import torch

from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.data import Collater
from pathlib import Path
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model


audio_wav_path, device, dtype = ...
audio_decoder = AudioDecoder(dtype=torch.float32, device=device)
fbank_converter = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=True,
    device=device,
    dtype=dtype,
)
collater = Collater(pad_value=1)

model = load_conformer_shaw_model("conformer_shaw", device=device, dtype=dtype)
model.eval()

with Path(audio_wav_path).open("rb") as fb:
    block = MemoryBlock(fb.read())

decoded_audio = audio_decoder(block)
src = collater(fbank_converter(decoded_audio))["fbank"]
seqs, padding_mask = get_seqs_and_padding_mask(src)

with torch.inference_mode():
  seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
  seqs, padding_mask = model.encoder(seqs, padding_mask)
```

## 评测 (Evaluation)

### SeamlessM4T 评测
如需复现论文基准结果，或在自定义测试集上使用相同指标进行评估，请查阅[说明文档](src/seamless_communication/cli/m4t/evaluate)。

### SeamlessExpressive 评测

以下为高效批量评测脚本：

```bash
export MODEL_DIR="/path/to/SeamlessExpressive/model"
export TEST_SET_TSV="input.tsv" # 您的 TSV 数据集文件，表头须包含 "id" 与 "audio"
export TGT_LANG="spa" # 目标翻译语言，可选包括 "fra"、"deu"、"eng"（"cmn" 与 "ita" 为实验性支持）
export OUTPUT_DIR="tmp/" # 生成的文本/离散单元/音频波形的输出目录
export TGT_TEXT_COL="tgt_text" # ${TEST_SET_TSV} 中用于计算 BLEU 分数的参考目标文本列名，该参数可省略
export DFACTOR="1.0" # 模型推理的持续时间因子，用于微调各位置预测的时长 (preddur=DFACTOR*preddur)，直接影响输出语速。数值越大语速越慢（默认 1.0）。详见表达力评测文档中关于持续时间因子的设置。
expressivity_evaluate ${TEST_SET_TSV} \
  --gated-model-dir ${MODEL_DIR} --task s2st --tgt_lang ${TGT_LANG} \
  --audio_root_dir "" --output_path ${OUTPUT_DIR} --ref_field ${TGT_TEXT_COL} \
  --model_name seamless_expressivity --vocoder_name vocoder_pretssel \
  --text_unk_blocking True --duration_factor ${DFACTOR}
```

详情请参阅[该文档小节](docs/expressive/README.md#automatic-evaluation)。

### SeamlessStreaming 与 Seamless 评测

[流式评测说明文档](src/seamless_communication/cli/streaming)提供了对 SeamlessStreaming 和 Seamless 模型进行基准评测的详细步骤。

## Unity.cpp
为实现无处不在的 Seamless Communication，我们开发了 unity.cpp，使用户能够在 GGML（轻量级 C 语言张量计算库）中高效运行 SeamlessM4T 模型，从而极大简化在各类边缘与资源受限平台上的集成与部署。

对给定音频进行转录/翻译：

```bash
./ggml/bin/unity --model seamlessM4T_medium.ggml input.wav
```

有关编译构建与更多使用方法，请参阅 [unity.cpp](ggml)。

## 表现力语音数据集 (Expressive Datasets)

我们构建了两个涵盖英语与另外五种语言（法语、德语、意大利语、普通话中文、西班牙语）之间富有表现力的语音到语音翻译数据集：mExpresso 和 mDRAL。目前我们已开源 mExpresso 从英语到其他语言方向的语音到文本（Speech-to-Text）数据，其余部分将很快陆续开源。详情请参阅 [README](docs/expressive/README.md#benchmark-datasets)。

### SeamlessAlignExpressive
我们引入了业界首个表现力语音对齐流程。从原始数据出发，表现力对齐流程能够自动挖掘不仅在语义上相同、而且整体情感与韵律表现力高度一致的音频片段对。为了展示这一方法，我们开源了基准数据集 SeamlessAlignExpressive 的元数据，用于验证对齐质量。SeamlessAlignExpressive 是首个用于表现力翻译的大规模（1.1 万+小时）多语言音频对齐集合。更多细节请参阅 [SeamlessAlignExpressive 说明文档](docs/expressive/seamless_align_expressive_README.md)。

## 音频转离散表示单元 (Units)
请参阅[此处说明文档](src/seamless_communication/cli/m4t/audio_to_units)。请注意：SeamlessM4T v1 模型采用归约单元（Reduced units），而其余模型均采用非归约离散单元（Non-reduced units）。

# 核心依赖库 (Libraries)

Seamless Communication 深度依托由 Meta 开发的 4 个开源核心库：

## [fairseq2](https://github.com/facebookresearch/fairseq2)
fairseq2 是我们面向序列建模的新一代开源算法库，为研究人员与开发者提供了用于机器翻译、语言建模以及各类序列生成任务的核心积木。本仓库中的所有 SeamlessM4T 模型均由 fairseq2 提供强劲底层驱动。

## [SONAR 与 BLASER 2.0](https://github.com/facebookresearch/SONAR)
SONAR（Sentence-level multimOdal and laNguage-Agnostic Representations，句子级多模态语言无关表征）是一种全新的多语言多模态句子嵌入空间，在 xsim 和 xsim++ 多语言相似度检索任务上的表现大幅超越 LASER3 和 LabSE 等现有嵌入模型。SONAR 为多种语言提供了高质量的文本与语音编码器。SeamlessAlign 数据集正是基于 SONAR 嵌入挖掘构建的。

BLASER 2.0 是我们最新的面向多模态翻译的基于模型的评估指标。作为 BLASER 的升级版本，它同时支持语音与文本。BLASER 2.0 直接在源信号上运算，因此不需要诸如 ASR-BLEU 等中间 ASR 语音识别系统的转录介入。与初代版本相同，BLASER 2.0 充分利用输入与输出句子嵌入之间的相似度，以 SONAR 作为核心嵌入空间。使用 BLASER 2.0 运行评测的脚本可从 [SONAR 仓库](https://github.com/facebookresearch/SONAR) 获取。

## [stopes](https://github.com/facebookresearch/stopes)
作为 Seamless Communication 项目的重要组成部分，我们对 stopes 库进行了深度拓展。版本 1 提供了用于构建翻译模型训练数据集的文本到文本挖掘工具；借助 SONAR，版本 2 扩展了对训练大规模语音翻译模型任务的支持。特别地，我们提供了读取/写入 fairseq audiozip 数据集的工具，以及支持语音到语音、文本到语音、语音到文本和文本到文本挖掘的全新挖掘流水线，所有功能均基于全新的 SONAR 嵌入空间构建。

## [SimulEval](https://github.com/facebookresearch/SimulEval)
SimulEval 是用于评测同声传译/同步翻译（Simultaneous Translation）模型的专业工具库。SimulEval 还提供了基于部分/增量输入以及灵活可扩展状态生成文本/语音的后端能力，用于实现流式低延迟推理。用户只需定义实现了 SimulEval 接口的 Agent，即可串联为完整的流水线。SeamlessStreaming 的内置 Agent 实现可在[此处](src/seamless_communication/streaming/agents)找到。

## [旧版存档] SeamlessM4T v1 使用指引
#### 微调 SeamlessM4T v1 模型
请查阅[说明文档](src/seamless_communication/cli/m4t/finetune)。

#### 端侧轻量化模型 (On-device models)
除 Seamless-M4T Large（2.3B）和 Medium（1.2B）模型外，我们还发布了一个专门面向边缘端侧推理的小型轻量模型（281M）。欲了解更多用法与模型架构细节，请参阅[端侧模型文档](docs/m4t/on_device_README.md)。

#### SeamlessAlign 多模态挖掘数据集
我们开源了 SeamlessAlign 的完整元数据，这是目前规模最大的多模态翻译开源数据集，共计包含超过 27 万小时对齐的语音与文本数据。社区可以根据 [SeamlessAlign 说明文档](docs/m4t/seamless_align_README.md) 重建完整数据集。

# 引用 (Citation)
如果您在学术研究中使用了 Seamless 或 Seamless 发布的任何模型、数据集或技术资产，请引用以下论文：

```bibtex
@inproceedings{seamless2023,
   title="Seamless: Multilingual Expressive and Streaming Speech Translation",
   author="{Seamless Communication}, Lo{\"i}c Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-juss{\`a}, Maha Elbayad, Hongyu Gong, Francisco Guzm{\'a}n, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, Mary Williamson",
  journal={ArXiv},
  year={2023}
}
```

# 开源协议 (License)

本项目包含三类协议许可：

以下非生成式组件遵循 [MIT 许可证](MIT_LICENSE)：
- [W2v-BERT 2.0 语音编码器](#w2v-bert-20-语音编码器)
- 源代码
- mExpresso 数据集中的仅文本（Text only）部分，详见 [SeamlessExpressive README](docs/expressive/README.md)。
- UnitY2 强制对齐提取器，详见 [UnitY2 对齐器文档](docs/m4t/unity2_aligner_README.md)。
- 配合 etox 数据集的语音毒性检测工具，详见 [ETOX README](src/seamless_communication/cli/toxicity/etox)。
- MuTox：通用多语言基于音频的毒性数据集与零样本检测器，详见 [Mutox README](src/seamless_communication/cli/toxicity/mutox)。

以下模型权重遵循 [CC-BY-NC 4.0 许可证](LICENSE)：
- SeamlessM4T 模型（v1 与 v2）。
- SeamlessStreaming 模型。

以下模型遵循 Seamless 专用许可协议，详见 [SEAMLESS_LICENSE](SEAMLESS_LICENSE)：
- Seamless 模型。
- SeamlessExpressive 模型。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月19日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
