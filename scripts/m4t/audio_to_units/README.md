# Convert raw audio into units (unit_extraction)

Raw audio needs to be converted to units to train UnitY models and vocoders. Units act as supervision for UnitY models, and are the input to the vocoders which synthesize speech from these units.

The unit extraction pipeline comprises the following steps:
- Compute features from layer 35 (determined empirically) of the pretrained XLSR v2 model ([paper](https://arxiv.org/abs/2111.09296)), which is a wav2vec2 model at the core.
- Assign features for each timestep to a collection of precomputed K-Means centroids to produce a sequence of units similar to extracting Hubert units as described in this [paper](https://arxiv.org/pdf/2107.05604.pdf).


## Quick start:
`audio_to_units` is run with the CLI, from the root directory of the repository.

```bash
m4t_audio_to_units <path_to_input_audio>
```

`audio_to_units` calls for `UnitExtractor` which provides a `predict` method to convert an audio to units.

The convenience method `resynthesize_audio` of `UnitExtractor`, can be used to resynthesize audio waveforms from units.
