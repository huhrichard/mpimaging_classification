from sklearn import model_selection
import pandas as pd
import os, fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


if __name__ == "__main__":
    patient_df = pd.read_csv("data/TMA_MPM_Summary_20191122.csv")
    deid = patient_df["Deidentifier patient number"].unique()
    img_name = patient_df['MPM image file per TMA core ']