import numpy as np
import torch

"""
upsample_scales -> upsample_rates
resblock_dilations -> resblock_dilation_sizes
in_channels -> model_in_dim
out_channels -> upsample_initial_channel
"""


def main():
    chkpt_root = "/checkpoint/mjhwang/experiments/231007-mel_vocoder-mls_multilingual_6lang/train_mls_multilingual_6lang_subset_hifigan.v1_8gpu_adapt"
    cfg = f"{chkpt_root}/config.yml"
    # TODO: display cfg
    chkpt = torch.load(f"{chkpt_root}/checkpoint-400000steps.pkl")
    del chkpt["model"]["discriminator"]
    conv_seq_map = {
        ".1.bias": ".bias",
        ".1.weight_g": ".weight_g",
        ".1.weight_v": ".weight_v",
    }

    def update_key(k):
        if k.startswith("input_conv"):
            k = k.replace("input_conv", "conv_pre")
        elif k.startswith("upsamples"):
            k = k.replace("upsamples", "ups")
            for _k, _v in conv_seq_map.items():
                k = k.replace(_k, _v)
        elif k.startswith("blocks"):
            k = k.replace("blocks", "resblocks")
            for _k, _v in conv_seq_map.items():
                k = k.replace(_k, _v)
        elif k.startswith("output_conv"):
            k = k.replace("output_conv", "conv_post")
            for _k, _v in conv_seq_map.items():
                k = k.replace(_k, _v)
        return k

    chkpt["model"] = {update_key(k): v for k, v in chkpt["model"]["generator"].items()}

    stats_path = f"{chkpt_root}/stats.npy"
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats[0].reshape(-1)).float()
    scale = torch.from_numpy(stats[1].reshape(-1)).float()
    chkpt["model"]["mean"] = mean
    chkpt["model"]["scale"] = scale

    for k in ["optimizer", "scheduler", "steps", "epochs"]:
        del chkpt[k]

    out_path = "/large_experiments/seamless/ust/changhan/checkpoints/fairseq2/pretssel_hifigan.pt"
    torch.save(chkpt, out_path)


if __name__ == "__main__":
    main()
