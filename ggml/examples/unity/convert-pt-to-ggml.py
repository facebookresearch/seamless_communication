# Convert UnitY model from PyTorch to ggml format
#
# Usage: python3.8 /private/home/dnn/ggml/ggml/examples/unity/convert-pt-to-ggml.py /large_experiments/seamless/ust/dnn/unity_large_audio_enc.pt /private/home/dnn/ggml/ggml/examples/unity/models/unity-large
# 
import io
import sys
import struct
import torch
import numpy as np
from pathlib import Path
from convert_pt_states import generate_mapping


if len(sys.argv) < 3:
    print("Usage: convert-pt-to-ggml.py model.pt dir-output [use-f32]\n")
    sys.exit(1)

fname_inp   = Path(sys.argv[1])
dir_out     = Path(sys.argv[2])

# try to load PyTorch binary data
try:
    model_bytes = open(fname_inp, "rb").read()
    with io.BytesIO(model_bytes) as fp:
        checkpoint = torch.load(fp, map_location="cpu")
except Exception:
    print("Error: failed to load PyTorch model file:" , fname_inp)
    sys.exit(1)

hparams = {"n_text_vocab": 256064, "n_audio_enc_dim": 1024, "n_audio_enc_ffn_dim": 4096, "n_audio_enc_feat_dim": 160, "n_audio_enc_layer": 24, "n_audio_enc_head": 16}
print("hparams:", hparams)

list_vars = checkpoint["model"]
state_map = generate_mapping(list_vars)

# output in the same directory as the model
fname_out = dir_out / "ggml-model.bin"

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 4:
    use_f16 = False
    fname_out = dir_out / "ggml-model-f32.bin"

fout = fname_out.open("wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
for key in hparams.keys():
    fout.write(struct.pack("i", hparams[key]))
fout.write(struct.pack("i", use_f16))

exclude_list = []
exclude_list += [f"encoder.w2v_encoder.w2v_model.encoder.layers.{i}.conv_module.batch_norm.num_batches_tracked" for i in range(24)]

for name in list_vars.keys():
    if list_vars[name] is None or name in exclude_list or "adaptor" in name:
        continue
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " , name ,  " with shape: ", data.shape)

    n_dims = len(data.shape)

    # TODO: Convert to fp16 when necessary!
    ftype = 0
    if name not in state_map:
        continue
    # header
    # if 'pos_bias' in name:
    #     import pdb; pdb.set_trace()
    #     print(data.shape)
    str_ = state_map[name].encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims-1-i]))
    fout.write(str_)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " , fname_out)
print("")

