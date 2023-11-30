# Evaluating SeamlessM4T models

Refer to the [SeamlessM4T README](../../../../../docs/m4t) for an overview of the M4T models.

Refer to the [inference README](../predict/README.md) for how to run inference with SeamlessM4T models.

## Quick start:
We use SACREBLEU library for computing BLEU scores and [JiWER library](https://github.com/jitsi/jiwer) is used to compute these CER and WER scores. 

Evaluation can be run with the CLI, from the root directory of the repository.

The model can be specified with `--model_name`: `seamlessM4T_v2_large` or `seamlessM4T_large` or `seamlessM4T_medium` 

```bash
m4t_evaluate --data_file <path_to_data_tsv_file> --task <task_name> --tgt_lang <tgt_lang> --output_path <path_to_save_evaluation_output> --ref_field <ref_field_name> --audio_root_dir <path_to_audio_root_directory>
```
## Note
1. We use raw (unnormalized) references to compute BLEU scores for S2TT, T2TT tasks.
2. For ASR task, src_lang needs to be passed as <tgt_lang> 
3. `--src_lang` arg needs to be specified to run evaluation for T2TT task

