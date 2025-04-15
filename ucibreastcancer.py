import folktexts
import pdb
import pandas as pd

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset
import os
from ucimlrepo import fetch_ucirepo 
import json

os.environ["OPENAI_API_KEY"] = json.loads("secrets.txt")["open_ai_key"]


descs = {
15:"""Samples arrive periodically as a Wisconson doctor reports his clinical cases related to breast cancer from 1992. 
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person."""}

names = {
15:"Breast Cancer Wisconsin",
}

testing_data = 15
name = names[testing_data]
description = descs[testing_data]

ucirepo = fetch_ucirepo(id=testing_data) 
X = ucirepo.data.features 
y = ucirepo.data.targets.replace(2, 0).replace(4, 1)
sel_vars = ucirepo.variables[['name', 'type', 'description']]

data = pd.concat([X, y], axis=1)


columns_map: dict[str, object] = {}

columns_map["Clump_thickness"] = ColumnToText(
    "Clump_thickness",
    short_description="Clump_thickness",
    value_map=lambda x: f"{x} from 1-10"
)

columns_map["Uniformity_of_cell_size"] = ColumnToText(
    "Uniformity_of_cell_size",
    short_description="Uniformity_of_cell_size",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Uniformity_of_cell_shape"] = ColumnToText(
    "Uniformity_of_cell_shape",
    short_description="Uniformity_of_cell_shape",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Marginal_adhesion"] = ColumnToText(
    "Marginal_adhesion",
    short_description="Marginal_adhesion",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Single_epithelial_cell_size"] = ColumnToText(
    "Single_epithelial_cell_size",
    short_description="Single_epithelial_cell_size",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Bare_nuclei"] = ColumnToText(
    "Bare_nuclei",
    short_description="Bare_nuclei",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Bland_chromatin"] = ColumnToText(
    "Bland_chromatin",
    short_description="Bland_chromatin",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Normal_nucleoli"] = ColumnToText(
    "Normal_nucleoli",
    short_description="Normal_nucleoli",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Mitoses"] = ColumnToText(
    "Mitoses",
    short_description="Mitoses",
 value_map=lambda x: f"{x} from 1-10"
)

columns_map["Class"] = ColumnToText(
    "Class",
    short_description="Class",
    value_map=lambda x: {2: 'benign', 4:'malignant'}[x],
)

reentry_qa = MultipleChoiceQA(
    column='Class',
    text="Is this sample benign or malignant?",
    choices=(
        Choice("Benign", 0),
        Choice("Malignant", 1),
    ),
)

model_name = "openai/gpt-4o-mini"


data = pd.concat([X, y], axis=1)
data.to_csv('check_data.csv')

outcome = 'Class'

reentry_task = TaskMetadata(
    name=name,
    description=description,
    features=[x for x in columns_map.keys() if x != outcome],
    target=outcome,
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
)

num_data = data.shape[0]
subsampling = min(5000 / num_data, 0.99)


reentry_dataset = Dataset(
    data=data,
    task=reentry_task,
    test_size=0.95,
    val_size=0,
    subsampling=subsampling,   # NOTE: Optional, for faster but noisier results!
)

all_tasks = {
    "reentry": [reentry_task, reentry_dataset]
}

for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
    RESULTS_DIR = f"uci{name}"
    bench.run(results_root_dir=RESULTS_DIR)
