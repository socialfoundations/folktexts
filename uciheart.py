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
import os
import json
os.environ["OPENAI_API_KEY"] = json.loads("secrets.txt")["open_ai_key"]


descs = {45:"""Presence of heart disease based on multiple indicators from 1988. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person."""}

names = {45:"Heart Disease"}


attributes_45 = variable_mappings = {
    "age": "Age in years",
    "sex": {
        "description": "Sex",
        "values": {
            0: "female",
            1: "male"
        }
    },
    "cp": {
        "description": "Chest pain type",
        "values": {
            1: "typical angina",
            2: "atypical angina",
            3: "non-anginal pain",
            4: "asymptomatic"
        }
    },
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": {
        "description": "Fasting blood sugar > 120 mg/dl",
        "values": {
            0: "false",
            1: "true"
        }
    },
    "restecg": {
        "description": "Resting electrocardiographic results",
        "values": {
            0: "normal",
            1: "ST-T wave abnormality",
            2: "left ventricular hypertrophy"
        }
    },
    "thalach": "Maximum heart rate achieved",
    "exang": {
        "description": "Exercise-induced angina",
        "values": {
            0: "no",
            1: "yes"
        }
    },
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": {
        "description": "Slope of peak exercise ST segment",
        "values": {
            1: "upsloping",
            2: "flat",
            3: "downsloping"
        }
    },
    "ca": "Number of major vessels colored by fluoroscopy (0–3)",
    "thal": {
        "description": "Thalassemia status",
        "values": {
            3: "normal",
            6: "fixed defect",
            7: "reversible defect"
        }
    },
    "num": {
        "description": "Diagnosis of heart disease",
        "values": {
            0: "< 50% diameter narrowing",
            1: "> 50% diameter narrowing"
        }
    }
    }
testing_data = 45
attributes_dict = {45:attributes_45}
name = names[testing_data]
description = descs[testing_data]

ucirepo = fetch_ucirepo(id=testing_data) 
X = ucirepo.data.features 
y = ucirepo.data.targets.replace(2, 1).replace(3, 1).replace(4, 1)
sel_vars = ucirepo.variables[['name', 'type', 'description']]

data = pd.concat([X, y], axis=1)

columns_map: dict[str, object] = {}


columns_map: dict[str, object] = {}
columns_map["age"] = ColumnToText(
    "age",
    short_description="age",
 value_map=lambda x: x
)

columns_map["sex"] = ColumnToText(
    "sex",
    short_description="sex",
    value_map=lambda x: attributes_dict[testing_data]["sex"]["values"][x],
)

columns_map["cp"] = ColumnToText(
    "cp",
    short_description="constrictive pericarditis",
    value_map=lambda x: attributes_dict[testing_data]["cp"]["values"][x],
)

columns_map["trestbps"] = ColumnToText(
    "trestbps",
    short_description="resting blood pressure (on hospital admission)",
 value_map=lambda x: x
)

columns_map["chol"] = ColumnToText(
    "chol",
    short_description="serum cholestoral",
 value_map=lambda x: x
)

columns_map["fbs"] = ColumnToText(
    "fbs",
    short_description="fasting blood sugar > 120 mg/dl",
    value_map=lambda x: attributes_dict[testing_data]["fbs"]["values"][x],
)

columns_map["restecg"] = ColumnToText(
    "restecg",
    short_description="resting ECG",
    value_map=lambda x: attributes_dict[testing_data]["restecg"]["values"][x],
)

columns_map["thalach"] = ColumnToText(
    "thalach",
    short_description="maximum heart rate achieved",
 value_map=lambda x: x
)

columns_map["exang"] = ColumnToText(
    "exang",
    short_description="exercise induced angina",
    value_map=lambda x: attributes_dict[testing_data]["exang"]["values"][x],
)

columns_map["oldpeak"] = ColumnToText(
    "oldpeak",
    short_description="ST depression induced by exercise relative to rest",
 value_map=lambda x: x
)

columns_map["slope"] = ColumnToText(
    "slope",
    short_description="ST segment/heart rate (ST/HR) slope",
    value_map=lambda x: attributes_dict[testing_data]["slope"]["values"][x],
)

columns_map["ca"] = ColumnToText(
    "ca",
    short_description="# major vessels colored by flourosopy (0-3)",
 value_map=lambda x: x
)

columns_map["thal"] = ColumnToText(
    "thal",
    short_description="thal",
    value_map=lambda x: attributes_dict[testing_data]["thal"]["values"][x],
)

columns_map["num"] = ColumnToText(
    "num",
    short_description="diagnosis of heart disease",
 value_map=lambda x: x
)



reentry_qa = MultipleChoiceQA(
    column='num',
    text="Does this person have heart disease?",
    choices=(
        Choice("No heart disease", 0),
        Choice("Yes, heart disease", 1),
    ),
)

model_name = "openai/gpt-4o-mini"


data = pd.concat([X, y], axis=1)
data.to_csv('check_data.csv')

outcome = 'num'




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
# asdfasdfasdf
