from sklearn import model_selection
import pandas as pd
import os, fnmatch
import numpy as np
import cv2
from skimage import feature
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def trainer_inner(model, inner_train_idx, inner_val_idx, X, Y)
    inner_X_train, inner_X_val = X[inner_train_idx], X[inner_val_idx]
    inner_Y_train, inner_Y_val = Y[inner_train_idx], Y[inner_val_idx]
    model.fit(inner_X_train, inner_Y_train)

    inner_Y_val_predict = model.predict_proba(inner_X_val)
    return inner_Y_val_predict


if __name__ == "__main__":
    base_datapath = "data/"
    patient_df = pd.read_csv(base_datapath + "TMA_MPM_Summary_20191122.csv")
    deids = np.array(patient_df["Deidentifier patient number"].unique())
    img_names = patient_df['MPM image file per TMA core ']
    img_feature_list = []
    img_resolution = 300
    y = patient_df[["BCR", "AP", "EPE"]]

    for img_name in img_names:
        img_path = find("*{}*".format(img_name), base_datapath)[0]
        bgr_img = cv2.imread(img_path)
        downscaled_img = cv2.resize(bgr_img, (img_resolution, img_resolution), cv2.INTER_LINEAR)
        hog_features = feature.hog(downscaled_img, orientations=16, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=False, multichannel=True)
        img_feature_list.append(hog_features)

    img_feature_list = np.array(img_feature_list).transpose(0,1)

    params_list = list(ParameterGrid({"ccp_alpha": np.linspace(0,1,5),
                                      "n_estimators": [10, 50, 100]}))

    inner_cv = model_selection.LeaveOneGroupOut()
    outer_cv = model_selection.LeaveOneGroupOut()

    for outer_train_idx, outer_test_idx in outer_cv.split(img_feature_list, y, groups=deids):
        outer_X_train, outer_X_test = img_feature_list[outer_train_idx], img_feature_list[outer_test_idx]
        outer_Y_train, outer_Y_test = y[outer_train_idx], y[outer_test_idx]
        outer_deids_train, outer_deids_test = deids[outer_train_idx], deids[outer_test_idx]
        for params in params_list:
            rfc = RandomForestClassifier(**params)
            for inner_train_idx, inner_val_idx in inner_cv.split(outer_X_train, outer_Y_train, outer_deids_train):









