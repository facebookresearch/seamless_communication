# On the Role of Speech Data in Reducing Toxicity Detection Bias

This dataset comprises additional annotations for the English and Spanish samples in the
MuTox test partition. To facilitate systematic evaluations of potential biases of speech-based
toxicity detection systems, 1954 samples have been annotated for mentions of demographic groups.
Annotators also corrected automated transcripts and adjusted judgments of toxicity where appropriate. 

These annotations form the basis of our paper comparing the performance and biases of speech- and text-based toxicity
detection systems, [available now on ArXiv](https://arxiv.org/abs/2411.08135). 

## License

These annotations are licensed under the MIT license (see the MIT_LICENSE file at the root of seamless_communication). 

## Annotations

* Annotations are made available for English and Spanish samples of the MuTox test set. 
* Annotations were produced by three annotators per language using an iterative, multi-stage process of annotation, review, and discussion.
* Annotators marked whether a category of group was mentioned, such as a gender identity, racial or ethic group, etc. 
* For each group category, annotators specified which specific demographic groups were mentioned. This was an open-ended free-text annotation.
* Group and group category annotations refer only to which groups are mentioned or referred to, and do not refer to the identity of the speaker. 
* Annotators also provided a new toxicity annotation, taking values `Yes`, `No`, `Cannot say` and `No consensus`.
* Finally, annotators marked whether the original ASR-produced transcript was correct, and if not, corrected it themselves.

### Group categories

Annotators were asked if samples mentioned demographic groups falling into one of the following categories:

* Gender identities
* Sexualities
* Religious groups
* Racial or ethnic groups
* Social classes or socio-economic statuses
* None of the above

Samples where annotators could not agree are marked as `No consensus`.

## Using the annotations

The annotations are available in this [TSV file](https://dl.fbaipublicfiles.com/seamless/datasets/mutox_group_annotations_v1.tsv).
Annotations can be joined with the original MuTox samples using the `id` column.

The columns are:
* `lang` specifies the language
* `transcript_is_correct` is whether the ASR transcript provided in the original MuTox dataset is correct
* `transcript_corrected` is the annotator-corrected transcript
* `contained_toxicity_corrected` is the annotator-corrected toxicity judgment
* `group_categories` is a list of categories of demographic groups mentioned in the sample, separated by '|', e.g. "Gender identities|Racial or ethnic groups"
* `groups` is a list of groups mentioned, separated by '|', e.g. "female, woman or girl|transgender"

## Citation

```bibtex 
@misc{bell2024,
      title={On the Role of Speech Data in Reducing Toxicity Detection Bias}
      author={Samuel J. Bell, Mariano Coria Meglioli, Megan Richards, Eduardo Sánchez, Christophe Ropers, Skyler Wang, Adina Williams, Levent Sagun, Marta R. Costa-jussà},
      year={2024},
      eprint={2411.08135},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



