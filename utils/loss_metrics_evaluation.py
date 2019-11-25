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

class performance_evaluation_cv(object):
    def __init__(self, nfold=5, total_epochs=50, multi_label=True, metrics_list=["auc", "f_max", "ap"],
                 ):
        self.metrics_list = metrics_list
        self.multi_label = multi_label
        self.nfold = nfold
        self.total_epochs = total_epochs

    def eval(self, predict, gt, states):
        performance_dict = {}
        for metrics in self.metrics_list:
            metric_func = globals()[metrics]
            performance_dict[metrics] = {}
            for state in states:
                performance_dict[metrics][state] = []
                for e in range(self.total_epochs):
                    if state == 'train':
                        metric_scores = []
                        for nth_fold in range(self.nfold):
                            # p = torch_tensor_np(predict[nth_fold],)
                            print(nth_fold, state, e)
                            p = predict[nth_fold][state][e]
                            g = gt[nth_fold][state][e]
                            metric_scores.append(metric_func(p, g))
                    else:
                        p = np.concatenate([predict[nth_fold][state][e] for nth_fold in range(self.nfold)], axis=0)
                        g = np.concatenate([gt[nth_fold][state][e] for nth_fold in range(self.nfold)], axis=0)
                        metric_scores = metric_func(p, g)

                    performance_dict[metrics][state].append(metric_scores)

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
    # print("recall: ", recall)
    # print("precision: ", precision)

    recall = np.clip(recall, a_min=eps, a_max=1)
    precision = np.clip(precision, a_min=eps, a_max=1)

    f_score = 2*(recall*precision)/(recall+precision)
    f_max = np.nanmax(f_score)
    max_threshold = threshold[np.nanargmax(f_score)]
    return f_max

def ap(predict, gt):
    return metrics.average_precision_score(y_score=predict,y_true=gt)

def balanced_acc_by_label(predict, gt):
    predict[predict>0.5] = 1
    predict[predict<=0.5] = 0
    p = predict.astype(int)
    g = gt.astype(int)
    score_list = []
    for idx in range(p.shape[1]):
        print("p:", p[:, idx])
        print("g:", g[:, idx])
        score_list.append(metrics.balanced_accuracy_score(y_pred=p[:, idx], y_true=g[:, idx]))
    return score_list
    # return metrics.(y_pred=p, y_true=g)

def f1_by_sample(predict, gt):
    predict[predict>0.5] = 1
    predict[predict<=0.5] = 0
    p = predict.astype(int)
    g = gt.astype(int)
    f1_score = 0
    for idx in range(p.shape[0]):
        f1_score += metrics.f1_score(y_pred=p[idx], y_true=g[idx])
    return f1_score/p.shape[0]

def f1_by_label(predict, gt):
    predict[predict>0.5] = 1
    predict[predict<=0.5] = 0
    p = predict.astype(int)
    g = gt.astype(int)
    score_list = []
    for idx in range(p.shape[1]):
        print("p:", p[:, idx])
        print("g:", g[:, idx])
        score_list.append(metrics.f1_score(y_pred=p[:, idx], y_true=g[:, idx]))
    return score_list


def evaluate_with_multi_label_classification(predict, gt, func):

    num_classes = predict.shape[-1]
    score_list = []
    for c in range(num_classes):
        score_list.append(func(y_pred=p[:, c], y_true=g[:, c]))

    return score_list

class loss_func(nn.Module):
    pass

