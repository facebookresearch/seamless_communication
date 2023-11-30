# Tool to compute toxicity in speech (ASR-ETOX) and text (ETOX)

In this tool, we combine an ASR model (M4T or whisper) + the ETOX toxicity detection tool
to compute a toxicity score for speech segments.

ETOX was developed as part of the NLLB project and provides a wordlist detection mechanism for 200 languages. By applying ASR on top of the ETOX detection, we can detect the toxicity in speech. You can find a description of the toxicity detection wordlists in the paper cited below.

## ASR-ETOX Usage

The script works by taking a TSV as input. The TSV needs a header with column names, it can have multiple columns. By defaut the script will look at the `audio` for the name of the audio file to load, this can be overriden with `--audio_column`.
The file path in the TSV can be absolute or relative to a root directory specified by `--audio_root_dir`. They can also be audiozip file formats with the appropriate byteoffset and length, e.g.: `fleurs_en_us_ogg_16khz.zip:89474600:49079`.

You can choose the ASR model to use, by default it will use `seamlessM4T_v2_large`. If you prefer to use [whisper](https://github.com/openai/whisper) you can specify a `--model_name` that starts with `whisper_` and finishes with the whisper model name (e.g. `whisper_large`).

## Outputs

The output of the script is a new TSV file with three columns:
- `text` the transcription
- `toxicity` the number of toxic words detected
- `bad_words` a list of toxic words, separated by `,`

## Sample Command

**ASR-ETOX**

- using M4T:
```bash
python -m seamless_communication.cli.toxicity.asr_etox --lang deu --audio_column ref_tgt_audio s2t/en-xx/deu.tsv ~/etox.tsv
```

- using Whisper:
```bash
python -m seamless_communication.cli.toxicity.asr_etox --model_name whisper_large --lang fra --audio_column ref_tgt_audio s2t/en-xx/fra.tsv ~/etox.test.tsv
```

**ETOX**

If you only care about getting the toxicity of text, you can use the etox.py script, with one text per line, specifying the language as the first argument.

```bash
cut -f 4 fleurs/s2t/en-xx/deu.tsv | python -m seamless_communication.cli.toxicity.etox deu > deu.toxicity.txt
```

You can also specify an input and output file:
```bash
python -m seamless_communication.cli.toxicity.etox deu deu.txt deu.toxicity.txt
```


# Citation
If you use ETOX, ASR-ETOX and SeamlessM4T in your work, please cite:


```bibtex
@misc{costajussà2023toxicity,
      title={Toxicity in Multilingual Machine Translation at Scale},
      author={Marta R. Costa-jussà and Eric Smith and Christophe Ropers and Daniel Licht and Jean Maillard and Javier Ferrando and Carlos Escolano},
      year={2023},
      eprint={2210.03070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

and

```bibtex
@article{seamlessm4t2023,
  title={SeamlessM4T—Massively Multilingual \& Multimodal Machine Translation},
  author={{Seamless Communication}, Lo\"{i}c Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye,  Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-juss\`{a} \footnotemark[3], Onur \,{C}elebi,Maha Elbayad,Cynthia Gao, Francisco Guzm\'an, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang},
  journal={ArXiv},
  year={2023}
}
```
