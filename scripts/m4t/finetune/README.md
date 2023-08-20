## Finetuning scripts for M4T

This section demonstrates an example of how M4T model can be finetuned for a subset of translation directions or modalities. 

Shared implementations of trainer and dataloader are not exhaustive. They were intentionally made simple in order to not obscure the specifics of data representation and optimization criteria during training.

## Data preparation

M4T training data is a multimodal parallel corpus. Each training sample has four parts: audio and text representation of a sample in source language, and corresponding audio and text representation of a sample in target language.

This kind of dataset can be prepared using `dataset.py` script that downloads FLEURS dataset from [HuggingFace datastes hub](https://huggingface.co/datasets/google/fleurs), extracts units from target audio samples and prepares a manifest consumable by `finetune.py`.

Example run command that prepares a training dataset for language pair English->Korean: 

```bash
python scripts/m4t/finetune/dataset.py \
 --source_lang eng \
 --target_lang kor \
 --split train \
 --save_dir /tmp
```
Path to the output manifest will be logged in the end of the command output: 

```bash
...
2023-08-19 03:23 INFO dataset - ..loaded 2600 source samples
2023-08-19 03:23 INFO dataset - Manifest saved to: /tmp/train_manifest.json
```

Manifest is a text file where each line represents information about a single dataset sample, serialized in JSON format.

## Finetuning

`finetune.py` is an example finetuning script that initializes dataloader, and launches a training loop with periodic evaluations on evaluation dataset. `torchrun` is the recommended way of launching it. 

Example launch command on a single node with 8 gpus:

```
torchrun \
   --rdzv-backend=c10d \
   --rdzv-endpoint=localhost:0 \
   --nnodes=1 \
   --nproc-per-node=8  \
 ./scripts/m4t/finetune/finetune.py \
   --train_dataset '<PATH TO TRAIN MANIFEST>' \
   --eval_dataset '<PATH TO EVAL MANIFEST>' \
   --model_name seamlessM4T_large \
   --save_model_to /tmp/checkpoint.pt
```

Example of a training log: 

```
...
2023-08-19 02:27:06,009 INFO -- trainer.1871488: Eval after 350 updates: loss=8.7876 best_loss=8.7876 patience_steps_left=3
2023-08-19 02:27:06,009 INFO -- trainer.1871488: Saving model
2023-08-19 02:27:31,100 INFO -- trainer.1871488: Epoch 007 / update 00360: train loss=16.3779 last lr=5.27E-08
2023-08-19 02:27:38,249 INFO -- trainer.1871488: Epoch 007 / update 00370: train loss=16.3482 last lr=5.20E-08
2023-08-19 02:27:45,164 INFO -- trainer.1871488: Epoch 007 / update 00380: train loss=16.4406 last lr=5.13E-08
2023-08-19 02:27:52,521 INFO -- trainer.1871488: Epoch 007 / update 00390: train loss=16.3556 last lr=5.06E-08
2023-08-19 02:27:59,300 INFO -- trainer.1871488: Epoch 007 / update 00400: train loss=16.3055 last lr=5.00E-08
2023-08-19 02:27:59,919 INFO -- trainer.1871488: Run evaluation
2023-08-19 02:28:12,761 INFO -- trainer.1871488: Eval after 400 updates: loss=8.7711 best_loss=8.7711 patience_steps_left=3
2023-08-19 02:28:12,762 INFO -- trainer.1871488: Saving model
...
```



