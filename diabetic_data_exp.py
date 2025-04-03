import folktexts
import pandas as pd

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset
import os

# os.chdir('/users/bryanwilder/Dropbox/llm_preds')

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
    26: "Transfer from residential care facility",
    27: "Transfer from ambulatory surgery center",
    28: "Transfer from rehabilitation facility including rehab units of a hospital",
    29: "Transfer from critical access hospital",
    30: "Transfer from federal health care facility",
    99: "Unknown/other"
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
    short_description="race of patient",
    value_map=lambda x: x
)

gender_col = ColumnToText(
    "gender",
    short_description="gender of patient",
    value_map=lambda x: x
)

age_col = ColumnToText(
    "age",
    short_description="age of patient",
    value_map=lambda x: f"{x} years"
)

weight_col = ColumnToText(
    "weight",
    short_description="weight of patient",
    value_map=lambda x: f"{x} lbs"
)

admission_id = ColumnToText(
    "admission_type_id",
    short_description="how patient was admitted",
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
    value_map=lambda x: x
)

num_lab_proc_col = ColumnToText(
    "num_lab_procedures",
    short_description="Number of lab tests performed during the encounter",
    value_map=lambda x: f"{x} procedures"
)

num_proc_col = ColumnToText(
    "num_procedures",
    short_description="Number of procedures (other than lab tests) performed during the encounter",
    value_map=lambda x: f"{x} procedures"
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
    value_map=lambda x: categorize_icd9(x)
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
    value_map=lambda x: f"{x} mg/dL"
)

a1c_col = ColumnToText(
    "A1Cresult",
    short_description="A1C level",
    value_map=lambda x: f"{x} %"
)

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
    value_map=lambda x: x
)

outcome = ColumnToText(
    "readmitted",
    short_description="whether patient was readmitted with 30 days",
    value_map=lambda x: 1 if x == "<30" else 0
)


reentry_numeric_qa = DirectNumericQA(
    column='readmitted',
    text=(
        "Was the patient readmitted within 30 days?"
    ),
)


reentry_qa = MultipleChoiceQA(
    column='readmitted',
    text="Was the patient readmitted within 30 days?",
    choices=(
        Choice("Yes, they were", 1),
        Choice("No, they were not", 0),
    ),
)

# shelter_qa = MultipleChoiceQA(
#     column='EMERG_SHLTR',
#     text="Will this person use a homeless shelter in the next year?",
#     choices=(
#         Choice("Yes, they will use a homeless shelter in the next year", 1),
#         Choice("No, they will not use a homeless shelter in the next year", 0),
#     ),
# )

# shelter_numeric_qa = DirectNumericQA(
#     column='ONE_YEAR_SHELTER',
#     text=(
#         "Will this person use a homeless shelter in the next year?"
#     ),
# )

# mhip_qa = MultipleChoiceQA(
#     column='MHIP',
#     text="Will this person have inpatient mental health treatment in the next year?",
#     choices=(
#         Choice("Yes, they will have inpatient mental health treatment in the next year", 1),
#         Choice("No, they will not have inpatient mental health treatment in the next year", 0),
#     ),
# )

# mhip_numeric_qa = DirectNumericQA(
#     column='MHIP',
#     text=(
#         "Will this person have inpatient mental health treatment in the next year?"
#     ),
# )

# ed_qa = MultipleChoiceQA(
#     column='FOUR_ER',
#     text="Will this person have at least four emergency department visits in the next year?",
#     choices=(
#         Choice("Yes, they will have at least four emergency department visits in the next year", 1),
#         Choice("No, they will not have at least four emergency department visits in the next year", 0),
#     ),
# )

# ed_numeric_qa = DirectNumericQA(
#     column='FOUR_ER',
#     text=(
#         "Will this person have at least four emergency department visits in the next year?"
#     ),
# )



# reentry_outcome_col = ColumnToText(
#     'JAIL',
#     short_description="reentry within one year",
#     question=reentry_qa,
# )

# shelter_outcome_col = ColumnToText(
#     'EMERG_SHLTR',
#     short_description="shelter useage within one year",
#     question=shelter_qa,
# )

# invol_outcome_col = ColumnToText(
#     'MHIP',
#     short_description="inpatient mental health treatment within one year",
#     question=mhip_qa,
# )

# mortality_outcome_col = ColumnToText(
#     'FOUR_ER',
#     short_description="at least four emergency department visits within one year",
#     question=ed_qa,
# )


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


# all_outcomes = ['JAIL', 'FOUR_ER', 'EMERG_SHLTR', 'MHIP']
all_outcomes = ["readmitted"]

reentry_task = TaskMetadata(
    name="readmission prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='readmitted',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

# shelter_task = TaskMetadata(
#     name="shelter prediction",
#     description=TASK_DESCRIPTION,
#     features=[x for x in columns_map.keys() if x not in all_outcomes],
#     target='EMERG_SHLTR',
#     cols_to_text=columns_map,
#     sensitive_attribute=None,
#     multiple_choice_qa=reentry_qa,
#     direct_numeric_qa=reentry_numeric_qa,
# )

# mhip_task = TaskMetadata(
#     name="mental health inpatient prediction",
#     description=TASK_DESCRIPTION,
#     features=[x for x in columns_map.keys() if x not in all_outcomes],
#     target='MHIP',
#     cols_to_text=columns_map,
#     sensitive_attribute=None,
#     multiple_choice_qa=mhip_qa,
#     direct_numeric_qa=mhip_numeric_qa,
# )

# ed_task = TaskMetadata(
#     name="emergency department prediction",
#     description=TASK_DESCRIPTION,
#     features=[x for x in columns_map.keys() if x not in all_outcomes],
#     target='FOUR_ER',
#     cols_to_text=columns_map,
#     sensitive_attribute=None,
#     multiple_choice_qa=ed_qa,
#     direct_numeric_qa=ed_numeric_qa,
# )

# shelter_task.use_numeric_qa = False
# reentry_task.use_numeric_qa = False
# mhip_task.use_numeric_qa = False
# ed_task.use_numeric_qa = False



data = pd.read_csv("data/diabetic_data.csv")
import numpy as np
data['readmitted'] = np.where(data['readmitted'] == '<30', 1, 0)
num_data = len(data)
# we want to sample 10k
subsampling = 50000 / num_data

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


model_name = "openai/gpt-4o-mini"
import os
import json
os.environ["OPENAI_API_KEY"] = json.loads("secrets.txt")["open_ai_key"]

for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "diabetes_readmission"
    bench.run(results_root_dir=RESULTS_DIR)



# llm_clf = WebAPILLMClassifier(model_name=model_name, task=shelter_task)
# bench = Benchmark(llm_clf=llm_clf, dataset=shelter_dataset)
# RESULTS_DIR = "res_shelter"
# bench.run(results_root_dir=RESULTS_DIR)
