# Dataset Card for _folktexts_ <!-- omit in toc -->

[Folktexts](https://github.com/socialfoundations/folktexts) is a suite of Q&A
datasets with natural outcome uncertainty, aimed at evaluating LLMs' calibration
on unrealizable tasks.

The *folktexts* datasets are derived from US Census data products.
Namely, the datasets made available here are derived from the
[2018 Public Use Microdata Sample](https://www.census.gov/programs-surveys/acs/microdata/documentation/2018.html)
(PUMS). Individual features are mapped to natural text using the respective
[codebook](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf).
Each task relates to predicting different individual
characteristics (e.g., income, employment) from a set of demographic features
(e.g., age, race, education, occupation).

Importantly, every task has natural outcome uncertainty. That is, in general,
the features describing each row do not uniquely determine the task's label.
For calibrated models to perform well on this task, the model must correctly
output nuanced scores between 0 and 1, instead of simply outputting discrete
labels 0 or 1.

Namely, we make available the following tasks in natural language Q&A format:
- `ACSIncome`: Predict whether a working adult earns above $50,000 yearly.
- `ACSEmployment`: Predict whether an adult is an employed civilian.
- `ACSPublicCoverage`: Predict individual public health insurance coverage.
- `ACSMobility`: Predict whether an individual changed address within the last year.
- `ACSTravelTime`: Predict whether an employed adult has a work commute time longer than 20 minutes.


These tasks follow the same naming and feature/target columns as the
[folktables](https://github.com/socialfoundations/folktables)
tabular datasets proposed by
[Ding et al. (2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf).
The folktables tabular datasets have seen prevalent use in the algorithmic
fairness and distribution shift communities. We make available natural language
Q&A versions of these tasks.

The datasets are made available in standard multiple-choice Q&A format (columns
`question`, `choices`, `answer`, `answer_key`, and `choice_question_prompt`), as
well as in numeric Q&A format (columns `numeric_question`,
`numeric_question_prompt`, and `label`).
The numeric prompting (also known as *verbalized prompting*) is known to improve
calibration of zero-shot LLM risk scores
[[Tian et al., EMNLP 2023](https://openreview.net/forum?id=g3faCfrwm7);
[Cruz et al., NeurIPS 2024](https://arxiv.org/pdf/2407.14614)].

**The accompanying [`folktexts` python package](https://github.com/socialfoundations/folktexts)
eases customization, evaluation, and benchmarking with these datasets.**

Table of contents:
- [Dataset Details](#dataset-details)
- [Uses](#uses)
- [Dataset Structure](#dataset-structure)
- [Dataset Creation](#dataset-creation)
- [Citation](#citation)
- [More Information](#more-information)


## Dataset Details

### Dataset Description <!-- omit in toc -->

- **Language(s) (NLP):** English
- **License:** Code is licensed under the [MIT license](https://github.com/socialfoundations/folktexts/blob/main/LICENSE); Data license is governed by the U.S. Census Bureau [terms of service](https://www.census.gov/data/developers/about/terms-of-service.html).

### Dataset Sources <!-- omit in toc -->

- **Repository:** https://github.com/socialfoundations/folktexts
- **Paper:** https://arxiv.org/pdf/2407.14614
- **Data source:** [2018 American Community Survey Public Use Microdata Sample](https://www.census.gov/programs-surveys/acs/microdata/documentation/2018.html)

## Uses

The datasets were originally used to evaluate LLMs' ability to produce
calibrated and accurate risk scores in the [Cruz et al. (2024)](https://arxiv.org/pdf/2407.14614) paper.

Other potential uses include evaluating the fairness of LLMs' decisions,
as individual rows feature protected demographic attributes such as `sex` and
`race`.


## Dataset Structure

**Description of dataset columns:**
- `id`: A unique row identifier.
- `description`: A textual description of an individual's features, following a bulleted-list format.
- `instruction`: The instruction used for zero-shot LLM prompting (should be pre-appended to the row description).
- `question`: A question relating to the task's target column.
- `choices`: A list of two answer options relating to the above question.
- `answer`: The correct answer from the above list of answer options.
- `answer_key`: The correct answer key; i.e., `A` for the first choice, or `B` for the second choice.
- `choice_question_prompt`: The full multiple-choice Q&A text string used for LLM prompting.
- `numeric_question`: A version of the question that prompts for a *numeric output* instead of a *discrete choice output*.
- `label`: The task's label. This is the correct output to the above numeric question.
- `numeric_question_prompt`: The full numeric Q&A text string used for LLM prompting.
- `<tabular-columns>`: All other columns correspond to the tabular features in this task. Each of these features will also appear in text form on the above description column.

The dataset was randomly split in `training`, `test`, and `validation` data,
following an 80%/10%/10% split.
Only the `test` split should be used to evaluate zero-shot LLM performance.
The `training` split can be used for fine-tuning, or for fitting traditional
supervised ML models on the tabular columns for metric baselines.
The `validation` split should be used for hyperparameter tuning, feature
engineering or any other model improvement loop.

## Dataset Creation

### Source Data <!-- omit in toc -->

The datasets are based on publicly available data from the American Community
Survey (ACS) Public Use Microdata Sample (PUMS), namely the
[2018 ACS 1-year PUMS files](https://www.census.gov/programs-surveys/acs/microdata/documentation.2018.html#list-tab-1370939201).

#### Data Collection and Processing <!-- omit in toc -->

The categorical values were mapped to meaningful natural language
representations using the `folktexts` package, which in turn uses the official
[ACS PUMS codebook](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf).
The data download and processing was aided by the `folktables` python package,
which in turn uses the official US Census web API.

#### Who are the source data producers? <!-- omit in toc -->

U.S. Census Bureau.

## Citation

If you find this useful in your research, please consider citing the following paper:

```bib
@inproceedings{
cruz2024evaluating,
title={Evaluating language models as risk scores},
author={Andr{\'e} F Cruz and Moritz Hardt and Celestine Mendler-D{\"u}nner},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=qrZxL3Bto9}
}
```

## More Information

More information is available in the [`folktexts`](https://github.com/socialfoundations/folktexts) package repository
and the [accompanying paper](https://arxiv.org/pdf/2407.14614).

### Dataset Card Authors <!-- omit in toc -->

[André F. Cruz](https://github.com/andrefcruz)
