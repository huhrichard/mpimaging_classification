from utils.common_library import *
import torch.nn as nn
import sklearn.metrics as metrics

eps = 1e-6

class performance_evaluation(object):
    def __init__(self, metrics_list=["auc", "f_max", "ap"]):
        self.metrics_list = metrics_list

    def eval(self, predict, gt):
        performance_dict = {}
        for metrics in self.metrics_list:
            metric_func = globals()[metrics]
            performance_dict[metrics] = metric_func(predict, gt)

        return performance_dict

def torch_tensor_np(tensor):
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    np_array = tensor.detach().numpy()
    if np_array.shape[-1] == 1:
        np_array = np.squeeze(np_array)
    return np_array

def auc(predict, gt):
    np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
    print(np_predict)
    print(np_gt)
    auc = metrics.roc_auc_score(y_score=np_predict, y_true=np_gt)
    return auc


def f_max(predict, gt):
    np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
    recall, precision, threshold = metrics.precision_recall_curve(probas_pred=np_predict,
                                                             y_true=np_gt)
    print("recall: ", recall)
    print("precision: ", precision)

    recall = np.clip(recall, a_min=eps, a_max=1)
    precision = np.clip(precision, a_min=eps, a_max=1)

    f_score = 2*(recall*precision)/(recall+precision)
    f_max = np.nanmax(f_score)
    max_threshold = threshold[np.nanargmax(f_score)]

    return f_max

def ap(predict, gt):
    np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
    ap = metrics.average_precision_score(y_score=np_predict,y_true=np_gt)
    return ap


class loss_func(nn.Module):
    pass

