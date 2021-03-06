from utils.common_library import *
import cvtorchvision
from torch.utils.data import Dataset, DataLoader
import pandas
import os, fnmatch
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut


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
        self.label_name = 'Gleason score for TMA core'
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

        sample = {'input': cv2.cvtColor(cv2.imread(self.img_path_list[idx]), cv2.COLOR_BGR2RGB),
                  'gt': torch.from_numpy(self.gt_list[idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample


# TODO: Load data by patient ID
class mpImage_sorted_by_patient_dataset(Dataset):
    def __init__(self, img_dir, multi_label_gt_path, img_suffix=None,
                 transform=None, skip_with_notes=True, included_gscore=True):
        """

        :param img_path:
        :param multi_label_gt_path:
        :param img_suffix:
        """

        self.multi_label_df = pandas.read_csv(multi_label_gt_path)
        self.patient_unique_deid_list = self.multi_label_df["Deidentifier patient number"].unique()

        self.label_name = ["BCR", "AP", "EPE"]
        # self.g_score
        # self.multi_label_gt_list = np.array(self.multi_label_df[self.label_name])

        self.patient_img_list = []
        self.patient_deid_list = []
        notes = "Notes"

        # self.img_df = pandas.read_csv(image_deidentify_path)
        self.gt_list = []

        for p_idx, patient_id in enumerate(self.patient_unique_deid_list):
            patient_entry = self.multi_label_df[self.multi_label_df["Deidentifier patient number"] == patient_id]
            img_files = patient_entry['MPM image file per TMA core ']
            notes_entry = patient_entry[notes]

            # print(img_files)
            # img_list = []
            for img_idx, img_file in enumerate(img_files):
                if notes_entry[img_idx] == '' and skip_with_notes:
                    path_list = find("{}*".format(img_file), img_dir)
                # prin t(path_list)
                    self.patient_img_list.append(path_list[0])
                    self.patient_deid_list.append(patient_id)
            # self.patient_img_list.append(img_list)
                    g_score = patient_entry[p_idx]['Gleason score for TMA core']
                    g_score[g_score!="Normal"] = 1
                    g_score[g_score=="Normal"] = 0
                    g_score = np.expand_dims(np.array(g_score).astype(float), axis=-1)
                    # print(g_score)
                    other_label = np.array(patient_entry[p_idx][self.label_name].astype(float))
                    if included_gscore:
                        self.gt_list.append(np.concatenate([g_score, other_label], axis=-1))
                        # print(self.gt_list[-1])
                    else:
                        self.gt_list.append(np.concatenate([other_label], axis=-1))
        self.patient_deid_list = np.expand_dims(np.array(self.patient_deid_list).astype(int), axis=-1)
        if included_gscore:
            self.label_name = ['Gleason score for TMA core'] + self.label_name
        self.transform = transform

    def __len__(self):
        return len(self.patient_unique_deid_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.patient_img_list[idx])
        sample = {'input': cv2.cvtColor(cv2.imread(self.patient_img_list[idx]), cv2.COLOR_BGR2RGB),
                  'gt': torch.from_numpy(self.gt_list[idx]),
                  'deid': torch.from_numpy(self.patient_deid_list[idx])}
        # print(sample["input"])

        if self.transform:
            sample = self.transform(sample)
        # sample['input'] = torch.stack(sample['input'], dim=0)

        return sample


class mpImage_sorted_by_patient_dataset_2(Dataset):
    def __init__(self, img_dir, multi_label_gt_path, img_suffix=None,
                 transform=None, skip_with_notes=True, included_gscore=True):
        """

        :param img_path:
        :param multi_label_gt_path:
        :param img_suffix:
        """

        self.multi_label_df = pandas.read_csv(multi_label_gt_path,  keep_default_na=False)
        # self.multi_label_df.fillna('')
        self.patient_unique_deid_list = self.multi_label_df["Deidentifier patient number"].unique()
        self.img_prefixes = self.multi_label_df["MPM image file per TMA core "]
        self.deids = self.multi_label_df["Deidentifier patient number"]
        self.notes = self.multi_label_df["Notes"]


        self.label_name = ["BCR", "AP", "EPE"]
        # self.g_score
        # self.multi_label_gt_list = np.array(self.multi_label_df[self.label_name])

        self.patient_img_list = []
        self.patient_deid_list = []
        self.row_idx_list = []
        self.gt_list = []
        # notes = "Notes"

        # self.img_df = pandas.read_csv(image_deidentify_path)
        self.gt_list = []

        for idx, img_prefix in enumerate(self.img_prefixes):
            if self.notes[idx] != "" and skip_with_notes:
                continue
            path_list = find("{}*".format(img_prefix), img_dir)
            # prin t(path_list)
            self.patient_img_list.append(path_list[0])
            self.patient_deid_list.append([self.deids[idx]])
            self.row_idx_list.append([idx])
            if self.multi_label_df['Gleason_Score'][idx] == "Normal":
                g_score = 0
            else:
                g_score = 1
            g_score = np.array([g_score]).astype(float)
            other_label = np.array(self.multi_label_df.loc[idx, self.label_name].astype(float))
            if included_gscore:
                self.gt_list.append(np.concatenate([g_score, other_label], axis=-1))
                # print(self.gt_list[-1])
            else:
                self.gt_list.append(np.concatenate([other_label], axis=-1))
        # self.patient_img_list = np.array(self.patient_img_list)
        self.patient_deid_list = np.array(self.patient_deid_list).astype(int)
        self.row_idx_list = np.array(self.row_idx_list).astype(int)

        if included_gscore:
            self.label_name = ['Gleason_Score'] + self.label_name
        self.transform = transform

    def __len__(self):
        return len(self.patient_img_list)

    def __getitem__(self, idx):
        # idx = list(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        # print(self.patient_img_list[idx])
        # print(idx, self.patient_img_list)
        sample = {'input': cv2.cvtColor(cv2.imread(self.patient_img_list[idx]), cv2.COLOR_BGR2RGB),
                  'gt': torch.from_numpy(self.gt_list[idx]),
                  'deid': torch.from_numpy(self.patient_deid_list[idx]),
                  'row_idx': torch.from_numpy(self.row_idx_list[idx])}
        # print(sample["input"])

        if self.transform:
            sample = self.transform(sample)
        # sample['input'] = torch.stack(sample['input'], dim=0)

        return sample

class mpImage_4C_sorted_by_patient_dataset(Dataset):
    def __init__(self, img_dir, multi_label_gt_path, img_suffix=None,
                 transform=None, skip_with_notes=True, included_gscore=True):
        """

        :param img_path:
        :param multi_label_gt_path:
        :param img_suffix:
        """

        self.multi_label_df = pandas.read_csv(multi_label_gt_path,  keep_default_na=False)
        # self.multi_label_df.fillna('')
        self.patient_unique_deid_list = self.multi_label_df["Deidentifier patient number"].unique()
        self.img_prefixes = self.multi_label_df["MPM image file per TMA core "]
        self.deids = self.multi_label_df["Deidentifier patient number"]
        self.notes = self.multi_label_df["Notes"]


        self.label_name = ["BCR", "AP", "EPE"]
        # self.g_score
        # self.multi_label_gt_list = np.array(self.multi_label_df[self.label_name])

        self.patient_img_list = []
        self.patient_deid_list = []
        self.row_idx_list = []
        self.gt_list = []
        # notes = "Notes"

        # self.img_df = pandas.read_csv(image_deidentify_path)
        self.gt_list = []

        for idx, img_prefix in enumerate(self.img_prefixes):
            if self.notes[idx] != "" and skip_with_notes:
                continue
            img_prefix_split = img_prefix.split(' ')
            img_list = []
            for c in range(1,5):
                img_list += find("{}*{}*{}*".format(img_prefix_split[0], c, img_prefix_split[1]),
                                 img_dir)
            # print(img_list)
            # path_list = find("{}*".format(), img_dir)
            # prin t(path_list)
            self.patient_img_list.append(img_list)
            self.patient_deid_list.append([self.deids[idx]])
            self.row_idx_list.append([idx])
            if self.multi_label_df['Gleason_Score'][idx] == "Normal":
                g_score = 0
            else:
                g_score = 1
            g_score = np.array([g_score]).astype(float)
            other_label = np.array(self.multi_label_df.loc[idx, self.label_name].astype(float))
            if included_gscore:
                self.gt_list.append(np.concatenate([g_score, other_label], axis=-1))
                # print(self.gt_list[-1])
            else:
                self.gt_list.append(np.concatenate([other_label], axis=-1))

        self.patient_deid_list = np.array(self.patient_deid_list).astype(int)
        self.row_idx_list = np.array(self.row_idx_list).astype(int)

        if included_gscore:
            self.label_name = ['Gleason_Score'] + self.label_name
        self.transform = transform

    def __len__(self):
        return len(self.patient_img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.patient_img_list[idx])
        # print(idx, self.patient_img_list)
        # for img in self.patient_img_list[idx]:
        #     print(img)
        #     temp = cv2.imread(img, cv2.IMREAD_ANYDEPTH)
        #     print(temp.shape)
        sample = {
                  'input': np.stack([(cv2.imread(img, cv2.IMREAD_ANYDEPTH)/256).astype('uint8') for img in self.patient_img_list[idx]],
                                    axis=-1),
                  # 'input': cv2.cvtColor(cv2.imread(self.patient_img_list[idx][0], cv2.IMREAD_ANYDEPTH).astype(float), cv2.COLOR_GRAY2BGR),
                  'gt': torch.from_numpy(self.gt_list[idx]),
                  'deid': torch.from_numpy(self.patient_deid_list[idx]),
                  'row_idx': torch.from_numpy(self.row_idx_list[idx])
                    }
        # print(sample["input"])

        if self.transform:
            sample = self.transform(sample)
        # sample['input'] = torch.stack(input for input in sample['input']], dim=-3)

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

def leave_one_out_cross_validation(len_data):
    loo = LeaveOneOut()
    cv_rand_idx = np.random.permutation(len_data)
    cv_split_list = list(loo.split(cv_rand_idx))

    return cv_split_list

def leave_one_patient_out_cross_validation(data_idx, patient_deid):
    print(data_idx.shape, patient_deid.shape)
    logo = LeaveOneGroupOut()
    # cv_rand_idx = np.random.permutation(len_data)
    cv_split_list = list(logo.split(X=data_idx, groups=patient_deid.squeeze()))

    return cv_split_list

def nfold_cross_validation(len_data, n_fold=5):
    kf = KFold(n_splits=n_fold, shuffle=True)
    cv_rand_idx = np.random.permutation(len_data)
    cv_split_list = list(kf.split(cv_rand_idx))
    return cv_split_list

