'''
Plan: For each column in the dataset, randomly pick n. Out of these n columns, make these the outcome variables, discretize them, and get average results.

Problems:
- what if each feature has a distinct number of options - if dataset A had many more options than dataset B, don't we just trivially expect lower scores from dataset A?
'''

import folktexts
import pandas as pd
import numpy as np
import os
import json
import pdb

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset
import os

task_name = "diabetes_readmission"
# read in data + basic attributes
data = pd.read_csv("data/diabetic_data.csv")
data['readmitted'] = np.where(data['readmitted'] == '<30', 1, 0)
num_data = len(data)
# we want to sample 10k
subsampling = (2000 / 0.95) / num_data
# if too many nulls, don't use the col as a proxy
null_thres = 0.7

discharge_disposition_map = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short-term hospital",
    3: "Discharged/transferred to skilled nursing facility (SNF)",
    4: "Discharged/transferred to intermediate care facility (ICF)",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left against medical advice",
    8: "Discharged/transferred to home under care of a home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to a swing bed",
    16: "Discharged/transferred/referred to another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "Null (or not mapped)",
    19: "Expired at home",
    20: "Expired in a medical facility",
    21: "Expired, place unknown",
    22: "Discharged/transferred to a rehab facility including rehab units of a hospital",
    23: "Discharged/transferred to a long-term care hospital",
    24: "Discharged/transferred to a nursing facility certified as a Medicare swing bed",
    25: "Discharged/transferred to another rehab facility",
    26: "Discharged/transferred to a critical access hospital",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred to a psychiatric hospital or psychiatric distinct part unit",
    29: "Discharged/transferred to a critical access hospital",
    30: "Discharged/transferred to another Type of Health Care Institution not defined elsewhere",
}

admission_source_map = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another healthcare facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal delivery",
    12: "Premature delivery",
    13: "Sick baby",
    14: "Extramural birth",
    15: "Transfer from another hospital for outpatient services",
    17: "Transfer from another healthcare facility for outpatient services",
    18: "Transfer from hospital inpatient",
    19: "Transfer from hospital outpatient",
    20: "Transfer from ambulatory surgery center",
    21: "Transfer from hospice",
    22: "Transfer from rehabilitation facility",
    23: "Transfer from long-term care hospital",
    24: "Transfer from psychiatric hospital or unit",
    25: "Transfer from intermediate care facility",
#     26: "Transfer from residential care facility",
#     27: "Transfer from ambulatory surgery center",
#     28: "Transfer from rehabilitation facility including rehab units of a hospital",
#     29: "Transfer from critical access hospital",
#     30: "Transfer from federal health care facility",
#     99: "Unknown/other"
}

payer_code_mapping = {
    'MC': 'Medicare',
    'MD': 'Medicaid',
    'BC': 'Blue Cross',
    'SP': 'Self-pay',
    'CM': 'Champus',
    'UN': 'United Healthcare',
    'DM': 'Department of Defense',
    'CP': 'Champus/Tricare',
    'PP': 'Private Insurance',
    'WC': "Worker's Compensation",
    'HM': 'HMO (Health Maintenance Organization)',
    'OG': 'Other Government',
    'PO': 'Other Private',
    'CH': 'ChampVA',
    'MP': 'Managed Care, Private',
    'OT': 'Other',
    'SI': 'Self-Insured',
    'FR': "Federal Government",
    '?': 'Unknown'
}


def categorize_icd9(code):
    """
    Categorizes an ICD-9 diagnosis code into a broader category based on the UCI Diabetes dataset.
    
    :param code: str, ICD-9 diagnosis code (can be a number or an 'E'/'V' code)
    :return: str, category name
    """
    diag_mapping = {
        range(1, 140): 'Infectious and parasitic diseases',
        range(140, 240): 'Neoplasms (cancers)',
        range(240, 250): 'Endocrine, nutritional, and metabolic diseases',
        '250': 'Diabetes mellitus',
        range(251, 280): 'Other endocrine disorders',
        range(280, 290): 'Diseases of the blood and blood-forming organs',
        range(290, 320): 'Mental disorders',
        range(320, 390): 'Diseases of the nervous system and sense organs',
        range(390, 460): 'Diseases of the circulatory system',
        range(460, 520): 'Diseases of the respiratory system',
        range(520, 580): 'Diseases of the digestive system',
        range(580, 630): 'Diseases of the genitourinary system',
        range(630, 680): 'Complications of pregnancy, childbirth, and the puerperium',
        range(680, 710): 'Diseases of the skin and subcutaneous tissue',
        range(710, 740): 'Diseases of the musculoskeletal system and connective tissue',
        range(740, 760): 'Congenital anomalies',
        range(760, 780): 'Certain conditions originating in the perinatal period',
        range(780, 800): 'Symptoms, signs, and ill-defined conditions',
        range(800, 1000): 'Injury and poisoning',
        'E': 'External causes of injury and poisoning',
        'V': 'Factors influencing health status and contact with health services',
        '?': 'Unknown or missing'
    }
    
    if not code or code == '?':
        return diag_mapping['?']
    
    code = str(code)
    
    # Handle E and V codes
    if code.startswith('E'):
        return diag_mapping['E']
    elif code.startswith('V'):
        return diag_mapping['V']
    
    # Try converting to an integer for range checking
    try:
        num_code = int(float(code))  # Convert decimal strings like '250.02' to 250
        for key_range, category in diag_mapping.items():
            if isinstance(key_range, range) and num_code in key_range:
                return category
        return 'Unknown'
    except ValueError:
        return 'Invalid Code'




TASK_DESCRIPTION = """\
The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. Information was extracted from the database for encounters that satisfied the following criteria.
(1)	It is an inpatient encounter (a hospital admission).
(2)	It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis.
(3)	The length of stay was at least 1 day and at most 14 days.
(4)	Laboratory tests were performed during the encounter.
(5)	Medications were administered during the encounter. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

race_col = ColumnToText(
    "race",
    short_description="race",
    value_map={
        x: x for x in set(data["race"].tolist())
    }
)

gender_col = ColumnToText(
    "gender",
    short_description="gender",
    value_map={
        x: x for x in set(data["gender"].tolist())
    })

age_col = ColumnToText(
    "age",
    short_description="age in years",
    value_map={
        x: f"{x} years" for x in set(data["age"].tolist())
    })

weight_col = ColumnToText(
    "weight",
    short_description="weight in lbs",
    value_map={
        x: x for x in set(data["weight"].tolist())
    }
)

admission_id = ColumnToText(
    "admission_type_id",
    short_description="method of admission",
    value_map={
        1:"Emergency",
        2:"Urgent",
        3:"Elective",
        4:"Newborn",
        5:"Not Available",
        6:"Null (or blank)",
        7:"Trauma Center",
        8:"Not Mapped"
    }
)


discharge_disposition_id_col = ColumnToText(
    "discharge_disposition_id",
    short_description="discharge information",
    value_map=discharge_disposition_map
)

admission_source_col = ColumnToText(
    "admission_source_id",
    short_description="admission circumstances",
    value_map=admission_source_map
)

time_hosp_col = ColumnToText(
    "time_in_hospital",
    short_description="Integer number of days between admission and discharge",
    value_map=lambda x: f"{x} days"
)

payer_code_col = ColumnToText(
    "payer_code",
    short_description="primary payer of expenses",
    value_map=payer_code_mapping
)

medical_specialty_col = ColumnToText(
    "medical_specialty",
    short_description="specialty of admitting physician",
    value_map={
        x: x for x in set(data["medical_specialty"].tolist())
    })

num_lab_proc_col = ColumnToText(
    "num_lab_procedures",
    short_description="Number of lab tests performed during the encounter",
    value_map=lambda x: f"{x} tests"
)

num_proc_col = ColumnToText(
    "num_procedures",
    short_description="Number of procedures (other than lab tests) performed during the encounter",
    value_map={
        x: x for x in set(data["num_procedures"].tolist())
    }
)

medications_col = ColumnToText(
    "num_medications",
    short_description="Number of distinct generic names administered during the encounter",
    value_map=lambda x: f"{x} medications"
)

outpatient_col = ColumnToText(
    "number_outpatient",
    short_description="Number of outpatient visits of the patient in the year preceding the encounter",
    value_map=lambda x: f"{x} visits"
)

emergency_col = ColumnToText(
    "number_emergency",
    short_description="Number of emergency visits of the patient in the year preceding the encounter",
    value_map=lambda x: f"{x} visits"
)

emergency_col = ColumnToText(
    "number_inpatient",
    short_description="Number of inpatient visits of the patient in the year preceding the encounter",
    value_map=lambda x: f"{x} visits"
)

diag_1_col = ColumnToText(
    "diag_1",
    short_description="primary diagnosis",
    value_map=lambda x: categorize_icd9(x) # NOTE not using a dict is fine because we'll never turn this into an MC question
)

diag_2_col = ColumnToText(
    "diag_2",
    short_description="secondary diagnosis",
    value_map=lambda x: categorize_icd9(x)
)

diag_3_col = ColumnToText(
    "diag_3",
    short_description="tertiary diagnosis",
    value_map=lambda x: categorize_icd9(x)
)

number_diag_col = ColumnToText(
    "number_diagnoses",
    short_description="Number of diagnoses entered to the system",
    value_map=lambda x: f"{x} diagnoses"
)

max_glu_serum_col = ColumnToText(
    "max_glu_serum",
    short_description="Max glucose serum",
    value_map={
        x: f"{x} mg / dL" for x in set(data["max_glu_serum"].tolist())
    }
)

a1c_col = ColumnToText(
    "A1Cresult",
    short_description="A1C level",
    value_map={
        x: f"{x} %" for x in set(data["A1Cresult"].tolist())
    })

metformin_col = ColumnToText(
    "metformin",
    short_description="metformin dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

repaglinide_col = ColumnToText(
    "repaglinide",
    short_description="repaglinide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

nateglinide_col = ColumnToText(
    "nateglinide",
    short_description="nateglinide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

chlorpropamide_col = ColumnToText(
    "chlorpropamide",
    short_description="chlorpropamide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

glimepiride_col = ColumnToText(
    "glimepiride",
    short_description="glimepiride dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

acetohexamide_col = ColumnToText(
    "acetohexamide",
    short_description="acetohexamide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

glipizide_col = ColumnToText(
    "glipizide",
    short_description="glipizide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

tolbutamide_col = ColumnToText(
    "tolbutamide",
    short_description="tolbutamide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

pioglitazone_col = ColumnToText(
    "pioglitazone",
    short_description="pioglitazone dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

rosiglitazone = ColumnToText(
    "rosiglitazone",
    short_description="rosiglitazone dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

acarbose = ColumnToText(
    "acarbose",
    short_description="acarbose dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

miglitol = ColumnToText(
    "miglitol",
    short_description="miglitol dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

troglitazone = ColumnToText(
    "troglitazone",
    short_description="troglitazone dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

tolazamide = ColumnToText(
    "tolazamide",
    short_description="tolazamide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

examide = ColumnToText(
    "examide",
    short_description="examide dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

citoglipton = ColumnToText(
    "citoglipton",
    short_description="citoglipton dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

insulin = ColumnToText(
    "insulin",
    short_description="insulin dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

glyburide_metformin = ColumnToText(
    "glyburide-metformin",
    short_description="glyburide-metformin dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

glipizide_metformin = ColumnToText(
    "glipizide-metformin",
    short_description="glipizide-metformin dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

glimepiride_pioglitazone = ColumnToText(
    "glimepiride-pioglitazone",
    short_description="glimepiride-pioglitazone dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

metformin_rosiglitazone = ColumnToText(
    "metformin-rosiglitazone",
    short_description="metformin-rosiglitazone dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

metformin_pioglitazone = ColumnToText(
    "metformin-pioglitazone",
    short_description="metformin-pioglitazone dosage change during encounter",
    value_map={
        "Down": "went down",
        "Up": "went up",
        "Steady": "stayed steady",
        "No": "not prescribed"
    }
)

change = ColumnToText(
    "change",
    short_description="change in diabetic medications (either dosage or generic name)",
    value_map={
        "Ch": "Change",
        "No": "No change"
    }
)

diabetesMed = ColumnToText(
    "diabetesMed",
    short_description="any diabetic medication prescribed",
    value_map={
        x: x for x in set(data["diabetesMed"].tolist())
    }
)


if __name__ == "__main__":

    # iterate through randomly selected columns (or possible all columns? tbd)
    # for each column calculate auc; take an average

    ban_cols = []

    for col in data.columns:
        # apply null threshold
        if data[col].isnull().mean() >= null_thres:
            ban_cols.append(col)
        # can't be monolithic
        if len(set(data[col].tolist())) < 2:
            ban_cols.append(col)
        # can't be imbalanced to the point where subsampling induces nan aucs
        if data[col].value_counts(normalize=True).iloc[0] >= 0.99:
            print(col)
            ban_cols.append(col)

    discretize_cols = ["time_in_hospital", "number_diagnoses", "num_lab_procedures",
                        "num_medications"] # numerical cols; need to convert into quartiles
    imbalanced_cols = ["num_procedures", "number_outpatient", "number_emergency", "number_inpatient"] # numerical, but also one option dominates so we need to make a binary classification
    all_columns_map: dict[str, object] = {
        col_mapper.name: col_mapper
        for col_mapper in globals().values()
        if isinstance(col_mapper, ColumnToText)
    }

    all_tasks = {}

    for col_name, col in all_columns_map.items():

        if col.name in ban_cols: continue # skip the ones that can't be discretized

        tmp_map = all_columns_map.copy()
        filtered_data = data[data[col.name].notnull()]

        options_set = list(set(data[col.name].tolist()))
        final_col_name = f"{col.name}_binary"

        if len(options_set) == 2: # already binary; use the 2 options to make a yes/no question
            positive, negative = max(options_set), min(options_set)
            filtered_data[final_col_name] = (filtered_data[col.name] == positive).astype(int) # positive -> 1; negative -> 0

            newCol = ColumnToText(
                final_col_name,
                short_description=col.short_description,
                value_map={
                    0: col.value_map(negative),
                    1: col.value_map(positive)
                }
            )
            tmp_map[final_col_name] = newCol

            numeric_q = DirectNumericQA(
                column=final_col_name,
                text=(
                    f"What is this person's {col.short_description}?"
                ),
            )
            
            mc_q = MultipleChoiceQA(
                column=final_col_name,
                text=f"What is the value of {col.short_description}?",
                choices=(
                    Choice(col.value_map(positive), 1),
                    Choice(col.value_map(negative), 0),
                ),
            )

        elif col.name in discretize_cols: # numerical value; use the median
            median = np.median(filtered_data[col.name])
            filtered_data[final_col_name] = (filtered_data[col.name] > median).astype(int) # if median above, say 1; otherwise 0
            
            newCol = ColumnToText(
                final_col_name,
                short_description=col.short_description,
                value_map={
                    0: f"less than or equal to {median}",
                    1: f"greater than {median}"
                }
            )
            tmp_map[final_col_name] = newCol

            numeric_q = DirectNumericQA(
                column=final_col_name,
                text=(
                    f"What is this person's {col.short_description}?"
                ),
            )
            
            mc_q = MultipleChoiceQA(
                column=final_col_name,
                text=f"Is the value of {col.short_description} above or below/equal to {col.value_map(median)}?",
                choices=(
                    Choice(f"greater than {col.value_map(median)}", 1),
                    Choice(f"less than or equal to {col.value_map(median)}", 0),
                ),
            )
        else: # categorical variable; take the mode and binarize this
            mode = filtered_data[col.name].mode().iloc[0]
            filtered_data[final_col_name] = (filtered_data[col.name] == mode).astype(int) # if equal to mode say 1; otherwise 0
            
            newCol = ColumnToText(
                final_col_name,
                short_description=col.short_description,
                value_map={
                    0: f"not equal to {mode}",
                    1: f"equal to {mode}"
                }
            )
            tmp_map[final_col_name] = newCol

            numeric_q = DirectNumericQA(
                column=final_col_name,
                text=(
                    f"What is this person's {col.short_description}?"
                ),
            )
            
            mc_q = MultipleChoiceQA(
                column=final_col_name,
                text=f"Is the value of {col.short_description} equal to {col.value_map(mode)}?",
                choices=(
                    Choice(f"equal to {col.value_map(mode)}", 1),
                    Choice(f"not equal to {col.value_map(mode)}", 0),
                ),
            )

        all_outcomes = list(set([final_col_name, col.name]))

        task = TaskMetadata(
            name=f"{final_col_name} prediction",
            description=TASK_DESCRIPTION,
            features=[x for x in all_columns_map.keys() if x not in all_outcomes],
            target=final_col_name,
            cols_to_text=tmp_map,
            sensitive_attribute=None,
            multiple_choice_qa=mc_q,
            direct_numeric_qa=numeric_q,
        )

        task.use_numeric_qa = False # TODO confirm this means we don't use the direct numeric question

        dataset = Dataset(
            data=filtered_data,
            task=task,
            test_size=.95,
            val_size=0,
            subsampling=subsampling,   # NOTE: Optional, for faster but noisier results!
        )

        all_tasks[final_col_name] = [task, dataset]

    model_name = "openai/gpt-4o-mini"
    with open("secrets.json", "r") as f:
        os.environ["OPENAI_API_KEY"] = json.load(f)["open_ai_key"]

    all_results = {}

    # pdb.set_trace()

    for taskname in all_tasks:
        task, dataset = all_tasks[taskname]
        llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
        llm_clf.set_inference_kwargs(batch_size=500)
        bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

        RESULTS_DIR = f"{task_name}/{taskname}"
        all_results[taskname] = bench.run(results_root_dir=RESULTS_DIR) # TODO figure out how to combine results
    
    avg_auc = np.mean([all_results[key]['roc_auc'] for key in all_results if all_results[key]['roc_auc'] is not np.nan])
    import pickle
    with open(f'{task_name}.pickle', 'wb') as handle:
        pickle.dump(all_results, handle)

    # New row to add
    new_row = {"dataset_name": {task_name}, "avg_auc": avg_auc} 

    # Check if file exists
    file_path = "dataset_results.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["dataset_name", "avg_auc"])

    # Append new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back to CSV
    df.to_csv(file_path, index=False)


    