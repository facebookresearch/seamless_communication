import re
from typing import Any, Dict, Mapping


def convert_model_state_dict(
    state_dict: Mapping[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a model state dictionary to fairseq2.

    :param state_dict:
        The original model state dictionary.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted model state dictionary that is compatible with fairseq2.
    """
    new_state_dict = {}

    def get_new_key(old_key: str) -> str:
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                return new_key

        return old_key

    # Convert module keys from fairseq to fairseq2.
    for old_key in state_dict.keys():
        new_key = get_new_key(old_key)

        new_state_dict[new_key] = state_dict[old_key]

    return new_state_dict



def convert_fairseq_checkpoint(
    checkpoint: Mapping[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a fairseq checkpoint to fairseq2.

    :param checkpoint:
        The original fairseq checkpoint.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted checkpoint that is compatible with fairseq2.
    """
    old_state_dict = checkpoint["model"]

    new_state_dict = convert_model_state_dict(old_state_dict, key_map)

    # We use the built-in version attribute of `torch.nn.Module`.
    try:
        del new_state_dict["encoder.version"]
    except KeyError:
        pass
    try:
        del new_state_dict["decoder.version"]
    except KeyError:
        pass

    try:
        del new_state_dict["encoder.embed_positions._float_tensor"]
    except KeyError:
        pass
    try:
        del new_state_dict["decoder.embed_positions._float_tensor"]
    except KeyError:
        pass

    return {"model": new_state_dict}