# Seamless - Speech to Speech and Speech to Text Metadata

This document contains metadata information for reconstructing the dataset we used for training our models.

## Format

The metadata format is similar to [NLLB bitext format](https://github.com/facebookresearch/LASER/tree/main/data/nllb200) with some small differences.

The metadata files are tab separated, gzip files. Each file corresponds to one alignment direction.

File naming convention:

- for text, we use 3 letters: e.g. `fra`, `eng`, `tur`
- for audio, we use 2 letters and a 'A': e.g. `frA`, `enA`, `trA`

For example, the direction `eng-trA` corresponds to information for reconstructing English text with Turkish speech alignments. Similarly, `enA-jpn` corresponds to "English speech with Japanese text", and `enA-frA` corresponds to "English speech with French speech".

Each line has 11 columns.

For Audio, the columns correspond to:

    - `cc_warc`: The warc file reference containing the public audio url
    - `cc_sha`: not used
    - `audio_speeh_segment_url`: space separated audio reference. See below.
    - `cc_lineno`: not used
    - `paragraph_digest`: not used
    - `sentence_digest`: not used
    - `text_lid_score`: not used
    - `laser_score`: score of the alignment
    - `direction`: direction, e.g. `enA-jpn`
    - `side`: side, e.g. `enA` or `jpn`
    - `line_no`: alignment number

`audio_speeh_segment_url` is a space separated audio reference. It has the following format:
`<url> <start_frame> <end_frame>`, where `start_frame` and `end_frame` correspond to the segment that needs to be extracted from the audio file that is referenced at `<url>`, resampled at 16000 Hz.

For text, the columns are similar to NLLB format (except being tab separated here):

- If the metadata comes from Common Crawl:

  - `cc_warc`: the reference to the Common Crawl WET file
  - `cc_sha`: the document sha1 in the WET file
  - `cc_document_url`: the url of the document referenced in the WET file
  - `cc_lineno`: the line number in the document referenced in the WET file
  - `paragraph_digest`: xxhash.xxh3_64_intdigest of the paragraph
  - `sentence_digest`: xxhash.xxh3_64_intdigest of the sentence
  - `text_lid_score`: language identification score, when available
  - `laser_score`: score of the alignment
  - `direction`: direction, e.g. `enA-jpn`
  - `side`: side, e.g. `enA` or `jpn`
  - `line_no`: alignment number

- If the metadata comes from other corpus:
  - `corpus`: corpus name
  - `cc_sha`: not used
  - `cc_document_url`: not used
  - `lineno`: line number in the document
  - `paragraph_digest`: xxhash.xxh3_64_intdigest of the paragraph
  - `sentence_digest`: xxhash.xxh3_64_intdigest of the sentence
  - `text_lid_score`: language identification score, when available
  - `laser_score`: score of the alignment
  - `direction`: direction, e.g. `enA-jpn`
  - `side`: side, e.g. `enA` or `jpn`
  - `line_no`: alignment number

## Data

[arb-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.arb-enA.tsv.gz) [ben-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.ben-enA.tsv.gz) [cat-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.cat-enA.tsv.gz) [dan-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.dan-enA.tsv.gz) [enA-est](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-est.tsv.gz) [enA-fin](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-fin.tsv.gz) [enA-jpn](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-jpn.tsv.gz) [enA-mlt](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-mlt.tsv.gz) [enA-nld](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-nld.tsv.gz) [enA-pol](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-pol.tsv.gz) [enA-por](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-por.tsv.gz) [enA-ron](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-ron.tsv.gz) [enA-slk](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-slk.tsv.gz) [enA-swe](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-swe.tsv.gz) [enA-swh](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-swh.tsv.gz) [enA-tur](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-tur.tsv.gz) [enA-ukr](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-ukr.tsv.gz) [enA-urd](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-urd.tsv.gz) [enA-vie](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-vie.tsv.gz) [arA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.arA-enA.tsv.gz) [arA-eng](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.arA-eng.tsv.gz) [beA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.beA-enA.tsv.gz) [caA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.caA-enA.tsv.gz) [caA-eng](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.caA-eng.tsv.gz) [csA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.csA-enA.tsv.gz) [csA-eng](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.csA-eng.tsv.gz) [cyA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.cyA-enA.tsv.gz) [cyA-eng](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.cyA-eng.tsv.gz) [daA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.daA-enA.tsv.gz) [daA-eng](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.daA-eng.tsv.gz) [deA-enA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.deA-enA.tsv.gz) [deA-eng](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.deA-eng.tsv.gz) [enA-esA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-esA.tsv.gz) [enA-fiA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-fiA.tsv.gz) [enA-frA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-frA.tsv.gz) [enA-hiA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-hiA.tsv.gz) [enA-idA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-idA.tsv.gz) [enA-itA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-itA.tsv.gz) [enA-knA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-knA.tsv.gz) [enA-koA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-koA.tsv.gz) [enA-mtA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-mtA.tsv.gz) [enA-nlA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-nlA.tsv.gz) [enA-plA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-plA.tsv.gz) [enA-ptA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-ptA.tsv.gz) [enA-rnA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-rnA.tsv.gz) [enA-ruA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-ruA.tsv.gz) [enA-skA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-skA.tsv.gz) [enA-svA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-svA.tsv.gz) [enA-swA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-swA.tsv.gz) [enA-taA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-taA.tsv.gz) [enA-teA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-teA.tsv.gz) [enA-tgA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-tgA.tsv.gz) [enA-thA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-thA.tsv.gz) [enA-trA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-trA.tsv.gz) [enA-ukA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-ukA.tsv.gz) [enA-urA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-urA.tsv.gz) [enA-uzA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-uzA.tsv.gz) [enA-viA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-viA.tsv.gz) [enA-zhA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.enA-zhA.tsv.gz) [eng-esA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-esA.tsv.gz) [eng-fiA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-fiA.tsv.gz) [eng-frA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-frA.tsv.gz) [eng-hiA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-hiA.tsv.gz) [eng-idA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-idA.tsv.gz) [eng-itA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-itA.tsv.gz) [eng-knA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-knA.tsv.gz) [eng-koA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-koA.tsv.gz) [eng-mtA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-mtA.tsv.gz) [eng-nlA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-nlA.tsv.gz) [eng-plA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-plA.tsv.gz) [eng-ptA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-ptA.tsv.gz) [eng-rnA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-rnA.tsv.gz) [eng-ruA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-ruA.tsv.gz) [eng-skA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-skA.tsv.gz) [eng-swA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-swA.tsv.gz) [eng-taA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-taA.tsv.gz) [eng-teA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-teA.tsv.gz) [eng-tgA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-tgA.tsv.gz) [eng-thA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-thA.tsv.gz) [eng-trA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-trA.tsv.gz) [eng-ukA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-ukA.tsv.gz) [eng-urA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-urA.tsv.gz) [eng-uzA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-uzA.tsv.gz) [eng-viA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-viA.tsv.gz) [eng-zhA](https://dl.fbaipublicfiles.com/seamless/data/seamless.dataset.metadata.public.eng-zhA.tsv.gz)

## Download script

You can use the `wet_lines` script to download and gather aligned text information from the metadata. This script can be found [here](https://github.com/kpu/preprocess/blob/wet/preprocess/wet_lines_main.cc).

### Example usage:

`zcat seamless.dataset.metadata.public.enA-swA.tsv.gz | egrep ^crawl-data | tr '\t' ' ' | wet_lines`

Based on metadata information it receives from stdin, wet_lines will download the corpora, find the paragraph and print the input with an additional column which corresponds to the text of the paragraph.

In order to retrieve the sentences from these paragraphs, one can use the sentence splitter available [here](https://github.com/facebookresearch/LASER/tree/main/utils). It will print the input (metadata + paragraph) with an additional column which corresponds to the text of the sentence.

### Reconstructing sentences from metadata:

`xzcat metadatafile.xz | egrep ^crawl-data | wet_lines | python -c "from sentence_cleaner_splitter.cleaner_splitter import *; split_clean()"`
