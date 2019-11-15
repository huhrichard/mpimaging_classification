from utils.common_library import *
import cvtorchvision
from torch.utils.data import Dataset, DataLoader
import pandas
import os, fnmatch


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

class mpImage_sorted_by_image_dataset(Dataset):
    def __init__(self, img_dir, gt_path, img_suffix=None, transform=None, skip_damaged=True):
        """

        :param img_path:
        :param gt_path:
        :param img_suffix:
        """

        self.df = pandas.read_csv(gt_path)
        self.img_dir = img_dir
        img_prefixes = self.df["MPM image file per TMA core "]
        self.img_path_list = []
        self.gt_list = []
        scores = self.df['Gleason score for TMA core']
        notes = self.df["Notes"]
        self.img_name = []
        for idx, img_prefix in enumerate(img_prefixes):
            # skip damaged image
            if notes[idx] == 'damaged' and skip_damaged is True:
                continue
            path_list = find("{}*".format(img_prefix), img_dir)
            # print(path_list)
            self.img_path_list.append(path_list[0])

            score = scores[idx]
            if score == "Normal":
                gt = np.zeros((1))
            else:
                gt = np.ones((1))
            self.gt_list.append(gt)

        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': cv2.imread(self.img_path_list[idx]),
                  'gt': torch.from_numpy(self.gt_list[idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample


# TODO: Load data by patient ID
class mpImage_sorted_by_patient_dataset(Dataset):
    def __init__(self, img_dir, image_label_path, multi_label_gt_path, img_suffix=None, transform=None, skip_damaged=True):
        """

        :param img_path:
        :param multi_label_gt_path:
        :param img_suffix:
        """

        self.multi_label_df = pandas.read_csv(multi_label_gt_path)
        self.patient_id_list = self.multi_label_df["DEIDENTIFIED"]

        self.label_name = ["BCR", "ap", "EPE"]
        self.multi_label_gt_list = self.multi_label_df[self.label_name]

        self.patient_img_list = []

        for idx, patient_id in enumerate():
            pass

        self.img_dir = img_dir
        img_prefixes = self.df["MPM image file per TMA core "]
        self.column_list = list(self.multi_label_df.columns)
        self.img_path_list = []
        self.gt_list = []
        scores = self.df['Gleason score for TMA core']
        notes = self.df["Notes"]
        self.img_name = []
        for idx, img_prefix in enumerate(img_prefixes):
            # skip damaged image
            if notes[idx] == 'damaged' and skip_damaged is True:
                continue
            path_list = find("{}*".format(img_prefix), img_dir)
            # print(path_list)
            self.img_path_list.append(path_list[0])

            score = scores[idx]
            if score == "Normal":
                gt = np.zeros((1))
            else:
                gt = np.ones((1))
            self.gt_list.append(gt)

        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': cv2.imread(self.img_path_list[idx]),
                  'gt': torch.from_numpy(self.gt_list[idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample


def cross_validation_and_test_split(len_data, n_folds=5, test_ratio=0.1, random_seed=None):
    np.random.seed(seed=random_seed)

    permuted_np_array = np.random.permutation(len_data)
    test_indices = permuted_np_array[:int(test_ratio*len_data)]
    cv_indices = permuted_np_array[int(test_ratio*len_data):]
    num_total_cv_indices = cv_indices.shape[0]
    n_folds_split_np_linspace = np.linspace(0, num_total_cv_indices, n_folds+1).astype(int)
    cv_split_nfolds = [cv_indices[n_folds_split_np_linspace[i]:n_folds_split_np_linspace[i+1]] for i in range(n_folds)]

    return cv_split_nfolds, test_indices

