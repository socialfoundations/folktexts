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

descs = {144:"""This dataset classifies people described by a set of attributes as \
good or bad credit risks in a classifier format. Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.""",
}

names = {144: "Statlog German Credit Data:",
}

attributes_144 = {
            "Attribute1": {
                "type": "qualitative",
                "name": "Status of existing checking account",
                "values": {
                    "A11": "< 0 DM",
                    "A12": "0 <= ... < 200 DM",
                    "A13": ">= 200 DM / salary assignments for at least 1 year",
                    "A14": "no checking account"
                }
            },
            "Attribute3": {
                "type": "qualitative",
                "name": "Credit history",
                "values": {
                    "A30": "no credits taken/ all credits paid back duly",
                    "A31": "all credits at this bank paid back duly",
                    "A32": "existing credits paid back duly till now",
                    "A33": "delay in paying off in the past",
                    "A34": "critical account/ other credits existing (not at this bank)"
                }
            },
            "Attribute4": {
                "type": "qualitative",
                "name": "Purpose",
                "values": {
                    "A40": "car (new)",
                    "A41": "car (used)",
                    "A42": "furniture/equipment",
                    "A43": "radio/television",
                    "A44": "domestic appliances",
                    "A45": "repairs",
                    "A46": "education",
                    "A47": "vacation (does not exist?)",
                    "A48": "retraining",
                    "A49": "business",
                    "A410": "others"
                }
            },
            "Attribute5": {
                "type": "numerical",
                "name": "Credit amount"
            },
            "Attribute6": {
                "type": "qualitative",
                "name": "Savings account/bonds",
                "values": {
                    "A61": "< 100 DM",
                    "A62": "100 <= ... < 500 DM",
                    "A63": "500 <= ... < 1000 DM",
                    "A64": ">= 1000 DM",
                    "A65": "unknown/ no savings account"
                }
            },
            "Attribute7": {
                "type": "qualitative",
                "name": "Present employment since",
                "values": {
                    "A71": "unemployed",
                    "A72": "< 1 year",
                    "A73": "1 <= ... < 4 years",
                    "A74": "4 <= ... < 7 years",
                    "A75": ">= 7 years"
                }
            },
            "Attribute8": {
                "type": "numerical",
                "name": "Installment rate in percentage of disposable income"
            },
            "Attribute9": {
                "type": "qualitative",
                "name": "Personal status and sex",
                "values": {
                    "A91": "male: divorced/separated",
                    "A92": "female: divorced/separated/married",
                    "A93": "male: single",
                    "A94": "male: married/widowed",
                    "A95": "female: single"
                }
            },
            "Attribute10": {
                "type": "qualitative",
                "name": "Other debtors / guarantors",
                "values": {
                    "A101": "none",
                    "A102": "co-applicant",
                    "A103": "guarantor"
                }
            },
            "Attribute11": {
                "type": "numerical",
                "name": "Present residence since"
            },
            "Attribute12": {
                "type": "qualitative",
                "name": "Property",
                "values": {
                    "A121": "real estate",
                    "A122": "building society savings agreement/life insurance",
                    "A123": "car or other, not in attribute 6",
                    "A124": "unknown / no property"
                }
            },
            "Attribute13": {
                "type": "numerical",
                "name": "Age in years"
            },
            "Attribute14": {
                "type": "qualitative",
                "name": "Other installment plans",
                "values": {
                    "A141": "bank",
                    "A142": "stores",
                    "A143": "none"
                }
            },
            "Attribute15": {
                "type": "qualitative",
                "name": "Housing",
                "values": {
                    "A151": "rent",
                    "A152": "own",
                    "A153": "for free"
                }
            },
            "Attribute16": {
                "type": "numerical",
                "name": "Number of existing credits at this bank"
            },
            "Attribute17": {
                "type": "qualitative",
                "name": "Job",
                "values": {
                    "A171": "unemployed/unskilled - non-resident",
                    "A172": "unskilled - resident",
                    "A173": "skilled employee / official",
                    "A174": "management/self-employed/highly qualified employee/officer"
                }
            },
            "Attribute19": {
                "type": "qualitative",
                "name": "Telephone",
                "values": {
                    "A191": "none",
                    "A192": "yes, registered under the customer's name"
                }
            },
            "Attribute20": {
                "type": "qualitative",
                "name": "Foreign worker",
                "values": {
                    "A201": "yes",
                    "A202": "no"
                }
            },
            "class": {
                "type": "qualitative",
                "name": "Credit Risk",
                "values": {
                    0: "Good",
                    1: "Bad",
                }
            }
        }

attributes_dict = {144:attributes_144}
testing_data = 144
name = names[testing_data]
description = descs[testing_data]

statlog_german_credit_data = fetch_ucirepo(id=testing_data) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets.replace(1, 0).replace(2, 1) 
sel_vars = statlog_german_credit_data.variables[['name', 'description']]

columns_map: dict[str, object] = {}
columns_map["Attribute1"] = ColumnToText(
    "Attribute1",
    short_description="Status of existing checking account",
    value_map=lambda x: attributes_dict[testing_data]["Attribute1"]["values"][x],
)

columns_map["Attribute2"] = ColumnToText(
    "Attribute2",
    short_description="Duration of checking account in months",
    value_map=lambda x: f"{x}",
)

columns_map["Attribute3"] = ColumnToText(
    "Attribute3",
    short_description="Credit history",
    value_map=lambda x: attributes_dict[testing_data]["Attribute3"]["values"][x],
)

columns_map["Attribute4"] = ColumnToText(
    "Attribute4",
    short_description="Purpose of account",
    value_map=lambda x: attributes_dict[testing_data]["Attribute4"]["values"][x],
)

columns_map["Attribute5"] = ColumnToText(
    "Attribute5",
    short_description="Credit amount (Germany 1994)",
    value_map=lambda x: f"{x}",
)

columns_map["Attribute6"] = ColumnToText(
    "Attribute6",
    short_description="Savings account/bonds",
    value_map=lambda x: attributes_dict[testing_data]["Attribute6"]["values"][x],
)

columns_map["Attribute7"] = ColumnToText(
    "Attribute7",
    short_description="Present employment since",
    value_map=lambda x: attributes_dict[testing_data]["Attribute7"]["values"][x],
)

columns_map["Attribute8"] = ColumnToText(
    "Attribute8",
    short_description="Installment rate in percentage of disposable income",
    value_map=lambda x: f"{x}%",
)

columns_map["Attribute9"] = ColumnToText(
    "Attribute9",
    short_description="Personal status and sex",
    value_map=lambda x: attributes_dict[testing_data]["Attribute9"]["values"][x],
)

columns_map["Attribute10"] = ColumnToText(
    "Attribute10",
    short_description="Other debtors / guarantors",
    value_map=lambda x: attributes_dict[testing_data]["Attribute10"]["values"][x],
)

columns_map["Attribute11"] = ColumnToText(
    "Attribute11",
    short_description="Present residence since",
    value_map=lambda x: f"{x}",
)

columns_map["Attribute12"] = ColumnToText(
    "Attribute12",
    short_description="Property",
    value_map=lambda x: attributes_dict[testing_data]["Attribute12"]["values"][x],
)

columns_map["Attribute13"] = ColumnToText(
    "Attribute13",
    short_description="Age",
    value_map=lambda x: f"{x}",
)

columns_map["Attribute14"] = ColumnToText(
    "Attribute14",
    short_description="Other installment plans",
    value_map=lambda x: attributes_dict[testing_data]["Attribute14"]["values"][x],
)

columns_map["Attribute15"] = ColumnToText(
    "Attribute15",
    short_description="Housing status",
    value_map=lambda x: attributes_dict[testing_data]["Attribute15"]["values"][x],
)


columns_map["Attribute16"] = ColumnToText(
    "Attribute16",
    short_description="Number of existing credits at this bank",
    value_map=lambda x: f"{x}",
)

columns_map["Attribute17"] = ColumnToText(
    "Attribute17",
    short_description="Job status",
    value_map=lambda x: attributes_dict[testing_data]["Attribute17"]["values"][x],
)

columns_map["Attribute18"] = ColumnToText(
    "Attribute18",
    short_description="Number of people being liable to provide maintenance for",
    value_map=lambda x: f"{x} people",
)

columns_map["Attribute19"] = ColumnToText(
    "Attribute19",
    short_description="Telephone",
    value_map=lambda x: attributes_dict[testing_data]["Attribute19"]["values"][x],
)

columns_map["Attribute20"] = ColumnToText(
    "Attribute20",
    short_description="foreign worker",
    value_map=lambda x: attributes_dict[testing_data]["Attribute20"]["values"][x],
)

columns_map["class"] = ColumnToText(
    "class",
    short_description="credit score risks",
    value_map=lambda x: attributes_dict[testing_data]["class"]["values"][x],
)

reentry_qa = MultipleChoiceQA(
    column='class',
    text="Does this person have a good credit or bad credit risk?",
    choices=(
        Choice("Yes, they have good credit", 0),
        Choice("No, they do not have good credit, they have bad credit", 1),
    ),
)

model_name = "openai/gpt-4o-mini"

proper_cols = list(sel_vars['name'].values)[:-1]

data = pd.concat([X[proper_cols], y], axis=1)
outcome = 'class'

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
