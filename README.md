# :book: folktexts   <!-- omit in toc -->

![Tests status](https://github.com/socialfoundations/folktexts/actions/workflows/python-tests.yml/badge.svg)
![PyPI status](https://github.com/socialfoundations/folktexts/actions/workflows/python-publish.yml/badge.svg)
![Documentation status](https://github.com/socialfoundations/folktexts/actions/workflows/python-docs.yml/badge.svg)
![PyPI version](https://badgen.net/pypi/v/folktexts)
![OSI license](https://badgen.net/pypi/license/folktexts)
![Python compatibility](https://badgen.net/pypi/python/folktexts)

Folktexts is a python package to evaluate and benchmark calibration of large
language models.
It enables using any transformers model as a classifier for tabular data tasks, 
and extracting risk score estimates from the model's output log-odds.

Several benchmark tasks are provided based on data from the American Community Survey.
Namely, each prediction task from the popular 
[folktables](https://github.com/socialfoundations/folktables) package is made available 
as a natural-language prompting task.

Package documentation can be found [here](https://socialfoundations.github.io/folktexts/).

**Table of contents:**
- [Installing](#installing)
- [Basic setup](#basic-setup)
- [Usage](#usage)
- [License and terms of use](#license-and-terms-of-use)


## Installing

Install package from [PyPI](https://pypi.org/project/folktexts/):

```
pip install folktexts
```

## Basic setup

1. Create condo environment

```
conda create -n folktexts python=3.11
conda activate folktexts
```

2. Install folktexts package

```
pip install folktexts
```

3. Create models dataset and results folder

```
mkdir results
mkdir models
mkdir datasets
```

3. Download transformers model and tokenizer into models folder

```
python -m folktexts.cli.download_models --model "google/gemma-2b" --save-dir models
```

4. Run benchmark

```
python -m folktexts.cli.run_acs_benchmark --results-dir results --data-dir datasets --task-name "ACSIncome" --model models/google--gemma-2b
```

Run `python -m folktexts.cli.run_acs_benchmark --help` to get a list of all
available benchmark flags.


## Usage

```py
from folktexts.acs import ACSDataset, ACSTaskMetadata
acs_task_name = "ACSIncome"

# Create an object that classifies data using an LLM
clf = LLMClassifier(
    model=model,
    tokenizer=tokenizer,
    task=ACSTaskMetadata.get_task(acs_task_name),
)

# Use a dataset or feed in your own data
dataset = ACSDataset(acs_task_name)

# Get risk score predictions out of the model
y_scores = clf.predict_proba(dataset)

# Optionally, can fit the threshold based on a small portion of the data
clf.fit(dataset[0:100])

# ...in order to get more accurate binary predictions
clf.predict(dataset)

# Compute a variety of evaluation metrics on calibration and accuracy
from folktexts.benchmark import CalibrationBenchmark
benchmark_results = CalibrationBenchmark(clf, dataset, results_dir="results").run()
```


## License and terms of use

Code licensed under the [MIT license](LICENSE).

The American Community Survey (ACS) Public Use Microdata Sample (PUMS) is
governed by the U.S. Census Bureau [terms of service](https://www.census.gov/data/developers/about/terms-of-service.html).
