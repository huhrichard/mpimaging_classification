from utils.common_library import *
import cvtorchvision
from torch.utils.data import Dataset, DataLoader

class mpImage_dataset(Dataset):
    def __init__(self, img_path, gt_path, img_suffix=None, transform=None):
        """

        :param img_path:
        :param gt_path:
        :param img_suffix:
        """

        self.img_list = img_path
        self.gt = gt_path
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': cv2.imread(self.img_list[idx]),
                  'output': self.gt[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def cross_validation_and_test_split(len_data, n_folds=5, test_ratio=0.1, random_seed=None):
    np.random.seed(seed=random_seed)

    permuted_np_array = np.random.permutation(len_data)
    test_indices = permuted_np_array[:int(test_ratio*len_data)]
    cv_indices = permuted_np_array[int(test_ratio*len_data):]
    num_total_cv_indices = cv_indices.shape[0]
    n_folds_split_np_linspace = np.linspace(0, num_total_cv_indices, n_folds+1)
    cv_split_nfolds = [cv_indices[n_folds_split_np_linspace[i]:n_folds_split_np_linspace[i+1]] for i in range(n_folds)]

    return cv_split_nfolds, test_indices

