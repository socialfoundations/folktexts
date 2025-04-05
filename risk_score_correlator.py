import argparse
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import json
import pdb

if __name__ == "__main__":
    folders = ["acsincome/openai/gpt-4o-mini_bench-3466366301", "acspubcov/openai/gpt-4o-mini_bench-3064965676", "acsunemployment/openai/gpt-4o-mini_bench-1236485182",
               "diabetes_readmission/openai/gpt-4o-mini_bench-2624672895", "ipums/openai/gpt-4o-mini_bench-2320147224", "sepsis/openai/gpt-4o-mini_bench-2353442754"]
    
    datapoints = []

    for folder in folders:
        
        # read in risk scores df, metrics json
        for f in os.listdir(folder):
            if f[-4:] == ".csv":
                risk_score_df = pd.read_csv(f"{folder}/{f}")
            if f[-5:] == ".json":
                with open(f"{folder}/{f}", "r") as fi:
                    metrics_json = json.load(fi)
        avg_risk = np.mean(risk_score_df["risk_score"])
        auc = metrics_json["roc_auc"]      

        # add average risk, metric to datapoints
        datapoints.append([avg_risk, auc])

    # plot datapoints
    plt.scatter([x[0] for x in datapoints], [x[1] for x in datapoints])
    plt.show()
