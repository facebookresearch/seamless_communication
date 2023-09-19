import torch
def map_state_key(pytorch_key, layer_idx=None):
    # Replace the layer index first
    if layer_idx is not None:
        pytorch_key = pytorch_key.replace(f".layers.{layer_idx}.", "/")
    
    # Replace common patterns in the state key
    translation_dict = {
        ".weight": "/w",
        ".bias": "/b",
        ".running_mean": "/m", # /running_mean doesn't work
        ".running_var": "/v",
        ".num_batches_tracked": "/n",
        "self_attn.": "self_attn_",
        "conv_module.": "conv_",
        "ffn1.": "ffn1_",
        "ffn2.": "ffn2_",
        "pos_conv.0": "pos_conv"
    }
    
    
    # Special mapping for pos_bias_u and pos_bias_v
    if "self_attn.pos_bias_u" in pytorch_key:
        pytorch_key = pytorch_key.replace("self_attn.pos_bias_u", "self_attn_pos_bias/u")
    elif "self_attn.pos_bias_v" in pytorch_key:
        pytorch_key = pytorch_key.replace("self_attn.pos_bias_v", "self_attn_pos_bias/v")
    for pytorch_pattern, model_pattern in translation_dict.items():
        pytorch_key = pytorch_key.replace(pytorch_pattern, model_pattern)
    
    # Replace the leading pattern and add layer index
    if layer_idx is not None:
        pytorch_key = pytorch_key.replace("encoder.w2v_encoder.w2v_model.encoder/", f"model/enc/h{layer_idx}/")
    else:
        pytorch_key = pytorch_key.replace("encoder.w2v_encoder.w2v_model.encoder.", f"model/enc/")
    pytorch_key = pytorch_key.replace("encoder.w2v_encoder.w2v_model.", f"model/")
    return pytorch_key


def generate_mapping(state_dict):
    mapping = {}
    for state in state_dict.keys():
        for layer_idx in range(24):
            if f".layers.{layer_idx}" in state:
                new_key = map_state_key(state, layer_idx)
                mapping[state] = new_key
        if "layers" not in state:
            mapping[state] = map_state_key(state)
    return mapping


# Testing
ckpt = torch.load('/large_experiments/seamless/ust/dnn/unity_large_audio_enc.pt')
state_dict = {}
for key in ckpt['model']:
    if ckpt['model'][key] is not None:
        state_dict[key] = ckpt['model'][key]

mapped_keys = generate_mapping(state_dict)
for old_key, new_key in mapped_keys.items():
    print(old_key, "=>", new_key)