## Finetuning scripts for M4T

This section demonstrates an example of M4T finetuning on a single translation direction: English-to-Korean.

The trainer and dataloader were designed mainly for demonstration purposes. Their simplicity should facilitate the code transparency and portability.

## Data preparation

M4T training dataset is a multimodal parallel corpus. Each training sample has four parts: audio and text representation of the sample in the source language, and its corresponding audio and text representation in the target language.

That kind of dataset can be prepared using `dataset.py` script that downloads FLEURS dataset from [HuggingFace datasets hub](https://huggingface.co/datasets/google/fleurs), (optionally) extracts units from the target audio samples, and prepares a manifest consumable by `finetune.py`. Manifest is a text file where each line represents information about a single dataset sample, serialized in JSON format.

List of input arguments for `dataset.py`:

```bash
  --source_lang SOURCE_LANG
                        M4T langcode of the dataset SOURCE language
  --target_lang TARGET_LANG
                        M4T langcode of the dataset TARGET language
  --split SPLIT         Dataset split/shard to download (`train`, `test`)
  --save_dir SAVE_DIR   Directory where the datasets will be stored with HuggingFace datasets cache files
```

Language codes should follow the notation adopted by M4T models.

Below is an example bash script that prepares a training and evaluation dataset for the translation direction English-to-Korean:

```bash
export DATASET_DIR=~/m4t_dataset
mkdir -p $DATASET_DIR

m4t_prepare_dataset \
  --source_lang eng \
  --target_lang kor \
  --split train \
  --save_dir $DATASET_DIR
m4t_prepare_dataset \
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
   --patience 3 \
   --model_name seamlessM4T_large \
   --save_model_to $DATASET_DIR/checkpoint.pt
```

Excerpt from an example finetuning log:

```
...
2023-08-21 14:46:16,936 INFO -- trainer.1100368: Eval after 300 updates: loss=8.7755 best_loss=8.7755 patience_steps_left=3
2023-08-21 14:46:16,936 INFO -- trainer.1100368: Saving model
2023-08-21 14:46:35,863 INFO -- trainer.1100368: Epoch 006 / update 00310: train loss=16.3768 last lr=5.68E-08
2023-08-21 14:46:42,610 INFO -- trainer.1100368: Epoch 006 / update 00320: train loss=16.3730 last lr=5.59E-08
2023-08-21 14:46:48,285 INFO -- trainer.1100368: Epoch 006 / update 00330: train loss=16.4598 last lr=5.50E-08
2023-08-21 14:46:54,390 INFO -- trainer.1100368: Epoch 006 / update 00340: train loss=16.4218 last lr=5.42E-08
2023-08-21 14:47:08,461 INFO -- trainer.1100368: Epoch 006 / update 00350: train loss=16.3906 last lr=5.35E-08
2023-08-21 14:47:09,067 INFO -- trainer.1100368: Run evaluation
2023-08-21 14:47:19,205 INFO -- trainer.1100368: Eval after 350 updates: loss=8.7462 best_loss=8.7462 patience_steps_left=3
2023-08-21 14:47:19,205 INFO -- trainer.1100368: Saving model
2023-08-21 14:47:44,981 INFO -- trainer.1100368: Epoch 007 / update 00360: train loss=16.4267 last lr=5.27E-08
2023-08-21 14:47:51,383 INFO -- trainer.1100368: Epoch 007 / update 00370: train loss=16.3630 last lr=5.20E-08
2023-08-21 14:47:58,305 INFO -- trainer.1100368: Epoch 007 / update 00380: train loss=16.3666 last lr=5.13E-08
2023-08-21 14:48:04,396 INFO -- trainer.1100368: Epoch 007 / update 00390: train loss=16.3605 last lr=5.06E-08
2023-08-21 14:48:10,630 INFO -- trainer.1100368: Epoch 007 / update 00400: train loss=16.3518 last lr=5.00E-08
...
```
