## Finetuning scripts for M4T

This section demonstrates an example of M4T finetuning on a single translation direction: English-to-Korean.

The trainer and dataloader were designed mainly for demonstration purposes. Their simplicity should facilitate the code transparency and portability.

## Data preparation

M4T training dataset is a multimodal parallel corpus. Each training sample has four parts: audio and text representation of the sample in the source language, and its corresponding audio and text representation in the target language.

That kind of dataset can be prepared using `dataset.py` script that downloads FLEURS dataset from [HuggingFace datasets hub](https://huggingface.co/datasets/google/fleurs), (optionally) extracts units from the target audio samples, and prepares a manifest consumable by `finetune.py`. Manifest is a text file where each line represents information about a single dataset sample, serialized in JSON format.

List of input arguments for `dataset.py`:

```bash
  --name NAME           HuggingFace name of the dataset to prepare.
  --source_lang SOURCE_LANG
                        M4T langcode of the dataset SOURCE language
  --target_lang TARGET_LANG
                        M4T langcode of the dataset TARGET language
  --split SPLIT         Dataset split/shard to download (`train`, `validation`, `test`)
  --save_dir SAVE_DIR   Directory where the datastets will be stored with HuggingFace datasets cache files
  --huggingface_token HUGGINGFACE_TOKEN
                        Your HuggingFace token, this is necessary for some datasets like GigaSpeech.
```

Language codes should follow the notation adopted by M4T models.

Below is an example bash script that prepares a training and evaluation dataset for the translation direction English-to-Korean:

```bash
export DATASET_DIR=~/m4t_dataset
mkdir -p $DATASET_DIR

m4t_prepare_dataset \
  --name google/fleurs \
  --source_lang eng \
  --target_lang kor \
  --split train \
  --save_dir $DATASET_DIR
m4t_prepare_dataset \
  --name google/fleurs \
  --source_lang eng \
  --target_lang kor \
  --split validation \
  --save_dir $DATASET_DIR
```


Output manifests will be stored in `${DATASET_DIR}/train_manifest.json` and `${DATASET_DIR}/validation_manifest.json`.


## Finetuning

`finetune.py` is an example finetuning script that initializes dataloaders, and launches training loop with periodic scoring against the validation dataset.
It is recommended to launch it with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). Multi-gpu and multi-node training are supported out of the box.

List of input arguments for `finetune.py`:

```bash
  --train_dataset TRAIN_DATASET
                        Path to manifest with train samples
  --eval_dataset EVAL_DATASET
                        Path to manifest with eval samples
  --model_name MODEL_NAME
                        Base model name (e.g, `seamlessM4T_medium`, `seamlessM4T_large`)
  --save_model_to SAVE_MODEL_TO
                        Path to save best finetuned model
  --seed SEED           Randomizer seed value
  --batch_size BATCH_SIZE
                        Batch size for training and evaluation
  --patience PATIENCE   Set early termination after `patience` number of evaluations without eval loss improvements
  --max_epochs MAX_EPOCHS
                        Max number of training epochs
  --learning_rate LEARNING_RATE
                        Finetuning learning rate
  --warmup_steps WARMUP_STEPS
                        Number of steps with linearly increasing learning rate
  --eval_steps EVAL_STEPS
                        Get eval loss after each `eval_steps` training steps
  --log_steps LOG_STEPS
                        Log inner loss after each `log_steps` training steps
  --mode {FinetuneMode.SPEECH_TO_SPEECH,FinetuneMode.SPEECH_TO_TEXT,FinetuneMode.TEXT_TO_SPEECH}
                        * `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model;
                        * `TEXT_TO_SPEECH` -- finetune only T2U;
                        * `SPEECH_TO_TEXT` -- finetune only S2T
```

The scripts supports three modes of finetuning:
- `SPEECH_TO_SPEECH`: in this case all model weights except the text encoder will be engaged;
- `TEXT_TO_SPEECH`: only text-to-unit part of the model will be engaged in the finetuning, other weights will be frozen;
- `SPEECH_TO_TEXT`: only speech-to-text part of the model will be engaged in the finetuning.

The referenced finetuning script does not support finetuning of the text encoder.


Below is an example bash script that launches finetuning of M4T-large on the dataset prepared earlier, using a single node with eight GPUs:

```
torchrun \
   --rdzv-backend=c10d \
   --rdzv-endpoint=localhost:0 \
   --nnodes=1 \
   --nproc-per-node=8  \
   --no-python \
  m4t_finetune \
   --mode SPEECH_TO_TEXT \
   --train_dataset $DATASET_DIR/train_manifest.json  \
   --eval_dataset $DATASET_DIR/validation_manifest.json \
   --learning_rate 1e-6 \
   --warmup_steps 100 \
   --max_epochs 10 \
   --patience 5 \
   --model_name seamlessM4T_v2_large \
   --save_model_to $DATASET_DIR/checkpoint.pt
```

Excerpt from an example finetuning log:

```
...
2024-01-17 03:13:12,608 INFO -- trainer: Eval after 200 updates: loss=4.5721 best_loss=4.4743 patience_steps_left=7
2024-01-17 03:13:19,859 INFO -- trainer: Epoch 004 / update 00210: train loss=4.4922 last lr=6.90E-07
2024-01-17 03:13:27,946 INFO -- trainer: Epoch 004 / update 00220: train loss=4.4694 last lr=6.74E-07
2024-01-17 03:13:36,320 INFO -- trainer: Epoch 004 / update 00230: train loss=4.4760 last lr=6.59E-07
2024-01-17 03:14:08,554 INFO -- trainer: Epoch 005 / update 00240: train loss=4.3438 last lr=6.45E-07
2024-01-17 03:14:16,529 INFO -- trainer: Epoch 005 / update 00250: train loss=4.2979 last lr=6.32E-07
2024-01-17 03:14:17,382 INFO -- trainer: Run evaluation
2024-01-17 03:14:31,172 INFO -- trainer: Eval after 250 updates: loss=4.4967 best_loss=4.4743 patience_steps_left=6
2024-01-17 03:14:38,497 INFO -- trainer: Epoch 005 / update 00260: train loss=4.2690 last lr=6.20E-07
2024-01-17 03:14:46,505 INFO -- trainer: Epoch 005 / update 00270: train loss=4.2489 last lr=6.09E-07
2024-01-17 03:14:54,796 INFO -- trainer: Epoch 005 / update 00280: train loss=4.2422 last lr=5.98E-07
2024-01-17 03:15:02,976 INFO -- trainer: Epoch 005 / update 00290: train loss=4.1874 last lr=5.87E-07
2024-01-17 03:15:34,510 INFO -- trainer: Epoch 006 / update 00300: train loss=4.1768 last lr=5.77E-07
2024-01-17 03:15:35,329 INFO -- trainer: Run evaluation
2024-01-17 03:15:49,634 INFO -- trainer: Eval after 300 updates: loss=4.4688 best_loss=4.4688 patience_steps_left=10
2024-01-17 03:15:49,634 INFO -- trainer: Saving model
2024-01-17 03:16:08,825 INFO -- trainer: Epoch 006 / update 00310: train loss=4.1509 last lr=5.68E-07
2024-01-17 03:16:16,979 INFO -- trainer: Epoch 006 / update 00320: train loss=4.0949 last lr=5.59E-07
2024-01-17 03:16:25,142 INFO -- trainer: Epoch 006 / update 00330: train loss=4.1053 last lr=5.50E-07
2024-01-17 03:16:32,966 INFO -- trainer: Epoch 006 / update 00340: train loss=4.1237 last lr=5.42E-07
2024-01-17 03:16:53,995 INFO -- trainer: Epoch 006 / update 00350: train loss=4.0980 last lr=5.35E-07
2024-01-17 03:16:54,690 INFO -- trainer: Run evaluation
2024-01-17 03:17:08,073 INFO -- trainer: Eval after 350 updates: loss=4.4463 best_loss=4.4463 patience_steps_left=10
2024-01-17 03:17:08,074 INFO -- trainer: Saving model
...
```
