from utils.common_library import *
import torch.nn as nn
import sklearn.metrics as metrics

eps = 1e-6

class performance_evaluation(object):
    def __init__(self, multi_label=True, metrics_list=["auc", "f_max", "ap"]):
        self.metrics_list = metrics_list
        self.multi_label = multi_label

    def eval(self, predict, gt):
        performance_dict = {}
        np_predict, np_gt = torch_tensor_np(predict), torch_tensor_np(gt)
        for metrics in self.metrics_list:
            metric_func = globals()[metrics]
            # if self.multi_label:
            #     performance_dict[metrics] = evaluate_with_multi_label_classification(np_predict,
            #                                                                          np_gt,
            #                                                                          metric_func)
            # else:
            performance_dict[metrics] = metric_func(np_predict, np_gt)
            print("{}: {}".format(metrics, performance_dict[metrics]))

        return performance_dict

class bcel_multi_output(nn.Module):
    def __init__(self):
        super(bcel_multi_output, self).__init__()
        pass

    def forward(self, predict, gt):
        if predict.shape != gt.shape:
            gt = gt.unsqueeze(-1).repeat(1,1,predict.shape[-1])
        return nn.functional.binary_cross_entropy(predict, gt)

def torch_tensor_np(tensor):
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    np_array = tensor.detach().numpy()
    # if np_array.shape[-1] == 1:
    #     np_array = np.squeeze(np_array)
    return np_array

def auc(predict, gt):
    return metrics.roc_auc_score(y_score=predict, y_true=gt)


def f_max(predict, gt):
    recall, precision, threshold = metrics.precision_recall_curve(probas_pred=predict,
                                                             y_true=gt)
    print("recall: ", recall)
    print("precision: ", precision)

    recall = np.clip(recall, a_min=eps, a_max=1)
    precision = np.clip(precision, a_min=eps, a_max=1)

    f_score = 2*(recall*precision)/(recall+precision)
    f_max = np.nanmax(f_score)
    max_threshold = threshold[np.nanargmax(f_score)]
    return f_max

def ap(predict, gt):
    return metrics.average_precision_score(y_score=predict,y_true=gt)

def f1(predict, gt):
    predict[predict>0.5] = 1
    predict[predict<=0.5] = 0
    return metrics.f1_score(y_true=gt.astype(int), y_pred=predict.astype(int))

def evaluate_with_multi_label_classification(predict, gt, func):

    num_classes = predict.shape[-1]
    score_list = []
    for c in range(num_classes):
        score_list.append(func(predict[:,c], gt[:, c]))

    return score_list

class loss_func(nn.Module):
    pass

