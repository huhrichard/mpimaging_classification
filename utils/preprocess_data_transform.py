from utils.common_library import *
from cvtorchvision import cvtransforms
import torchvision.transforms as transforms


class compose_input_output_transform(object):
    def __init__(self, input_transform, output_transform=None, with_gt = True):
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.with_gt = with_gt

    def __call__(self, sample):
        sample['input'] = self.input_transform(sample['input'])
        if self.with_gt:
            sample["output"] = self.output_transform(sample['output'])

        return sample



