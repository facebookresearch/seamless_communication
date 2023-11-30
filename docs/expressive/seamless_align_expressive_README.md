# SeamlessAlignExpressive

Building upon our past work with WikiMatrix, CCMatrix, NLLB, SpeechMatrix and SeamlessM4T, we’re introducing the first expressive speech alignment procedure. Starting with raw data, the expressive alignment procedure automatically discovers pairs of audio segments sharing not only the same meaning, but the same overall expressivity. To showcase this procedure, we are making metadata available to create a benchmarking dataset called SeamlessAlignExpressive, that can be used to validate the quality of our alignment method. SeamlessAlignExpressive is the first large-scale collection of multilingual audio alignments for expressive translation for benchmarking.

## Format

The metadata files are space separated, gzip files. Each file corresponds to one alignment direction. File naming convention: we use 2 letters with an 'A': e.g. `frA`, `enA`, `deA`.

For example, the direction `deA-enA` corresponds to information for reconstructing German speech to English speech alignments.

Each line has 9 columns.

The columns correspond to:
    - `direction`: direction, e.g. `enA-deA`
    - `side`: side, e.g. `enA` or `deA`
    - `line_no`: alignment number
    - `cc_warc`: The public CC warc file reference containing the public audio url
    - `duration`: original file duration
    - `audio_speech_segment_url`: public audio reference
    - `audio_speech_start_frame`: start frame when the audio is resampled at 16kHz
    - `audio_speech_end_frame`: end frame when the audio is resampled at 16kHz
    - `laser_score`: score of the alignment


## Data

[deA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless_align_expressive/seamless.dataset.metadata.public.deA-enA.tsv.gz) [enA-esA](https://dl.fbaipublicfiles.com/seamless/data/seamless_align_expressive/seamless.dataset.metadata.public.enA-esA.tsv.gz) [enA-frA](https://dl.fbaipublicfiles.com/seamless/data/seamless_align_expressive/seamless.dataset.metadata.public.enA-frA.tsv.gz) [enA-itA](https://dl.fbaipublicfiles.com/seamless/data/seamless_align_expressive/seamless.dataset.metadata.public.enA-itA.tsv.gz) [enA-zhA](https://dl.fbaipublicfiles.com/seamless/data/seamless_align_expressive/seamless.dataset.metadata.public.enA-zhA.tsv.gz)
