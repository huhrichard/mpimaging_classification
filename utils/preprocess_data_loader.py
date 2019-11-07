import cv2
import torch
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


        sample = {'img': img,
                  'gt':gt}

        if self.transform:
            sample = self.transform(sample)

        return sample