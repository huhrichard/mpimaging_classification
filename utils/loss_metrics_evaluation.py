from utils.common_library import *
import torch.nn as nn

class performance_evaluation(object):
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list

    def eval(self, predict, gt):
        performance_dict = {}
        for metrics in self.metrics_list:
            performance_dict[metrics] = globals()[metrics](predict, gt)

        return performance_dict


def area_under_curve(predict, gt):
    pass


def f_max(predict, gt):
    pass

class loss_func(nn.Module):
    pass