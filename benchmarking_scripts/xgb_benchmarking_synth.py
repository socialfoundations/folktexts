from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
from skopt import BayesSearchCV
import pandas as pd
import os
import time
from sklearn.metrics import roc_auc_score
import numpy as np

if __name__ == "__main__":
    datasets = ["acsincome_synth.csv", "acspubcov_synth.csv", "acsunemployment_synth.csv", "ipums_synth.csv", "sepsis_synth.csv", "diabetic_data_synth.csv"][-3:-2]
    label_cols = ["PINCP", "PUBCOV", "ESR", "facility", "SepsisLabel", "readmitted"][-3:-2]

    for dataset_name, label_col in zip(datasets, label_cols):
        dataset = pd.read_csv(f"data/{dataset_name}")
        if "Unnamed: 0" in dataset.columns:
            dataset = dataset.drop(columns=['Unnamed: 0'])

        dataset[dataset.select_dtypes(include=['object']).columns] = dataset.select_dtypes(include=['object']).astype('category')
        # import pdb; pdb.set_trace()
        y_train = dataset[label_col]
        x_train = dataset.drop(columns=label_col)

        # if dataset_name == "acspubcov_synth.csv": # for pubcov, 1 = yes, 2 = no; map to 1/0
        #     y_train = -1 * y_train + 2
        # if dataset_name == "diabetic_data_synth.csv":
        #     y_train = y_train.map(lambda x: 1 if x == "<30" else 0)
        # if dataset_name == "ipums_synth.csv":
        #     x_train = x_train.drop(columns=["place_delivery", "wealth_index"])
            # import pdb; pdb.set_trace()


        # fit xgb on the x data
        # bayes_search = BayesSearchCV(
        #     XGBClassifier(enable_categorical=True, random_state=34),
        #     {
        #         'max_depth': (2, 10),
        #         'learning_rate': (1e-4, 0.15),
        #         'n_estimators': (50, 5000),
        #         'gamma': (0, 5),
        #         'min_child_weight': (0,5)
        #     },
        #     n_iter=20,
        #     scoring='neg_root_mean_squared_error',
        #     cv=3,
        #     # verbose=1,
        #     random_state=33
        # )

        # start_time = time.time()
        # bayes_search.fit(x_train, y_train)
        # end_time = time.time()
        # print(f"Time to optimize: {end_time - start_time}")

        # print(bayes_search.best_params_)

        # pre_augmentation_model = XGBClassifier(**bayes_search.best_params_, enable_categorical=True, random_state=34)
        pre_augmentation_model = XGBClassifier(enable_categorical=True, random_state=34)
        pre_augmentation_model.fit(x_train, y_train)
        
        # evaluate on x_data: accuracy and auc
        train_pred, train_label = pre_augmentation_model.predict(x_train), y_train.to_numpy()
        naive_acc = max(np.sum(train_label), len(train_label) - np.sum(train_label)) / len(train_label)
        train_acc = np.mean(train_pred == train_label)
        train_auc = roc_auc_score(train_label, pre_augmentation_model.predict_proba(x_train)[:,1])
        train_bas = balanced_accuracy_score(train_label, train_pred)

        # write to csv
        # schema: "dataset_name", "auc", "accuracy", "naive_acc"

        new_row = pd.DataFrame([[dataset_name, train_acc, train_bas, train_auc, naive_acc]], columns=['dataset_name', 'train_acc', 'train_balanced_accuracy', 'train_auc', 'naive_acc'])

        import pdb
        pdb.set_trace()
        output_csv_path = "benchmark_metrics_synth.csv"
        if not os.path.exists(output_csv_path):
            # write new row to csv directly
            new_row.to_csv(output_csv_path)
        else:
            old = pd.read_csv(output_csv_path)
            # add new row to existing csv
            new = pd.concat([old, new_row], ignore_index=True)[["dataset_name", "train_acc", "train_balanced_accuracy", "train_auc", "naive_acc"]]

            # write combined into csv
            new.to_csv(output_csv_path)