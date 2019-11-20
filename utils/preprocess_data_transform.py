from utils.common_library import *
from cvtorchvision import cvtransforms
import torchvision.transforms as transforms


class compose_input_output_transform(object):
    def __init__(self, input_transform, output_transform=None, with_gt = True):
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.with_gt = with_gt

    def __call__(self, sample):
        input = sample['input']
        # print(input)
        if self.input_transform is not None:
            if type(sample['input']) == list:
                # print(input.s)
                sample['input'] = [self.input_transform(s) for s in input]
            else:
                sample['input'] = self.input_transform(input)
        if self.with_gt:
            if self.output_transform is not None:
                sample["gt"] = self.output_transform(sample['gt'])

        return sample




