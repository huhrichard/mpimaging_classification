from utils.common_library import *
import torch.nn as nn
import sklearn.metrics as metrics

class performance_evaluation(object):
    def __init__(self, metrics_list=["auc", "f_max", "ap"]):
        self.metrics_list = metrics_list

    def eval(self, predict, gt):
        performance_dict = {}
        for metrics in self.metrics_list:
            performance_dict[metrics] = globals()[metrics](predict, gt)

        return performance_dict

def torch_tensor_np(tensor):
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    return tensor.numpy()

def auc(predict, gt):
    np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
    auc = metrics.roc_auc_score(y_score=np_predict, y_true=np_gt)
    return auc


def f_max(predict, gt):
    np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
    recall, precision, threshold = metrics.precision_recall_curve(pos_label=np_predict,
                                                             y_true=np_gt)

    f_score = 2*(recall*precision)/(recall+precision)
    f_max = np.max(f_score)
    max_threshold = threshold[np.argmax(f_score)]

    return f_max

def ap(predict, gt):
    np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
    ap = metrics.average_precision_score(y_score=np_predict,y_true=np_gt)
    return ap


class loss_func(nn.Module):
    pass

