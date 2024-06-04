# :book: folktexts   <!-- omit in toc -->
> :construction: Package under construction

![Tests status](https://github.com/AndreFCruz/folktexts/actions/workflows/python-tests.yml/badge.svg)
![PyPI status](https://github.com/AndreFCruz/folktexts/actions/workflows/python-publish.yml/badge.svg)
![Documentation status](https://github.com/AndreFCruz/folktexts/actions/workflows/python-docs.yml/badge.svg)
![PyPI version](https://badgen.net/pypi/v/folktexts)
![OSI license](https://badgen.net/pypi/license/folktexts)
![Python compatibility](https://badgen.net/pypi/python/folktexts)

Repo to host the `folktexts` project.

Package documentation can be found [here](https://andrefcruz.github.io/folktexts/)!

**Table of contents:**
- [Installing](#installing)
- [Usage](#usage)
- [License and terms of use](#license-and-terms-of-use)


## Installing

Install package from [PyPI](https://pypi.org/project/folktexts/):

```
pip install folktexts
```

## Developer mode

1. Create condo environment

```
conda create -n folktexts python=3.11      
conda activate folktexts
```

2. clone git repo

```
git clone git@github.com:AndreFCruz/folktexts.git  
```

2. Install folk texts package in edit mode

```
cd folktexts
pip install -e .
```

3. Create models dataset and results folder

```
mkdir results
mkdir models
mkdir datasets
```

3. Download models into models folder
(make sure you have Huggingface permission)

```
python -m folktexts.cli.download_models --model "google/gemma-2b" --save-dir models
```

4. Run benchmark
(It downloads dataset and runs benchmarking)

```
python folktexts/cli/run_llm_as_classifier.py --results-dir results --data-dir datasets --acs-task-name "ACSIncome" --model models/google--gemma-2b --subsampling 0.01
```

NOTE: saves results to results folder, not a subfolder!


## Usage

*This is a template of how we envision the API -- not yet fully implemented!*

```py
from folktexts.datasets import ACSDataset
from folktexts.acs import acs_income_task
from folktexts.qa_interface import BinaryQA

# Create an object that classifies data using an LLM
clf = LLMClassifier(
    model=model,
    tokenizer=tokenizer,
    task=acs_income_task,   # NOTE: the task should know how to map itself to text!
    qa_interface=BinaryQA,  # How to frame the question and extract outputs from the model
)

# Use a dataset or feed in your own data
dataset = ACSDataset(name="ACSIncome")

# Get risk score predictions out of the model
y_scores = clf.predict_proba(dataset)

# Optionally, can fit the threshold based on a small portion of the data
clf.fit(dataset[0:100])

# ...in order to get more accurate binary predictions
clf.predict(dataset)

# Compute a variety of evaluation metrics on calibration and accuracy
benchmark_results = run_llm_as_clf_benchmark(clf, dataset)
```



## License and terms of use

Code licensed under the [MIT license](LICENSE).

The American Community Survey (ACS) Public Use Microdata Sample (PUMS) is
governed by the U.S. Census Bureau [terms of service](https://www.census.gov/data/developers/about/terms-of-service.html).
