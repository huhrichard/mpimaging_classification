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
            performance_dict[metrics] = metric_func(np_gt, np_predict)
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
        prediction_list_by_state = {}
        gt_list_by_state = {}


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
                            # print(nth_fold, state, e)
                            p = predict[nth_fold][state][e]
                            g = gt[nth_fold][state][e]

                            metric_scores.append(metric_func(g, p))

                        # metric_scores = np.array(metric_scores)
                    else:
                        p = np.concatenate([predict[nth_fold][state][e] for nth_fold in range(self.nfold)], axis=0)
                        g = np.concatenate([gt[nth_fold][state][e] for nth_fold in range(self.nfold)], axis=0)
                        metric_scores = metric_func(g, p)

                    performance_dict[metrics][state].append(metric_scores)
                performance_dict[metrics][state] = np.array(performance_dict[metrics][state])

        return performance_dict

class bcel_multi_output(nn.Module):
    def __init__(self):
        super(bcel_multi_output, self).__init__()
        pass

    def forward(self, predict, gt):
        if predict.shape != gt.shape:
            gt = gt.unsqueeze(-1).repeat(1,1,predict.shape[-1])
        return nn.functional.binary_cross_entropy(predict, gt)

class bin_focal_loss_multi_output(nn.Module):
    def __init__(self,
                 alpha=0.5,
                 gamma=2):
        super(bin_focal_loss_multi_output, self).__init__()
        self.alpha = torch.Tensor([alpha]).squeeze(-1)
        self.gamma = torch.Tensor([gamma]).squeeze(-1)

    def forward(self, predict, gt):
        self.alpha = self.alpha.to(predict.device)
        self.gamma = self.gamma.to(predict.device)
        if predict.shape != gt.shape:
            gt = gt.unsqueeze(-1).repeat(1,1,predict.shape[-1])
        fl = torch.Tensor([0])
        for idx, a in enumerate(self.alpha):
            fl += 0
        return self.focal_loss(predict, gt)

    def focal_loss(self, predict, gt):
        return self.alpha*(torch.mean((1-predict)**self.gamma)*torch.log(gt))

class multi_label_loss(nn.Module):
    def __init__(self, loss_function='BCE',
                 alpha=0.5,
                 gamma=2):
        super(multi_label_loss, self).__init__()
        self.alpha = torch.Tensor([alpha]).squeeze(-1)
        self.gamma = torch.Tensor([gamma]).squeeze(-1)
        self.loss_function = loss_function

    def forward(self, predict, gt):
        if predict.shape != gt.shape:
            gt = gt.unsqueeze(-1).repeat(1, 1, predict.shape[-1])
        if self.loss_function == 'BCE':
            return nn.functional.binary_cross_entropy(predict, gt)
        elif self.loss_function == 'FL':
            self.alpha = self.alpha.to(predict.device)
            self.gamma = self.gamma.to(predict.device)
            fl = self.focal_loss(predict, gt) + self.focal_loss(1-predict, 1-gt)
            print(fl)
            return fl.squeeze()

    def focal_loss(self, predict, gt):
        return -1*self.alpha*gt*(torch.mean(((1-predict)**self.gamma)*torch.log(predict)))

def torch_tensor_np(tensor):
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    np_array = tensor.detach().numpy()
    # if np_array.shape[-1] == 1:
    #     np_array = np.squeeze(np_array)
    return np_array

def auc(gt, predict):
    return metrics.roc_auc_score(gt, predict)


def f_max(gt, predict):
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
    return metrics.average_precision_score(gt, predict)

def balanced_acc_by_label(predict, gt):
    p = np.ones_like(predict)
    p[predict<=0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    score_list = []
    for idx in range(p.shape[1]):
        # print("p:", p[:, idx])
        # print("g:", g[:, idx])
        score_list.append(metrics.balanced_accuracy_score(g[:, idx], p[:, idx]))
    return score_list
    # return metrics.(y_pred=p, y_true=g)

def f1_by_sample(gt, predict):
    p = np.ones_like(predict)
    p[predict <= 0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    f1_score = 0
    for idx in range(p.shape[0]):
        f1_score += metrics.f1_score(g[idx], p[idx])
    return f1_score/p.shape[0]

def f1_by_label(gt, predict):
    p = np.ones_like(predict)
    p[predict <= 0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    return evaluate_with_multi_label_classification(g, p, metrics.f1_score)



def auc_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, metrics.roc_auc_score)

def ap_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, metrics.average_precision_score)

def fmax_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, f_max)

def evaluate_with_multi_label_classification(g, p, func):

    num_classes = p.shape[-1]
    score_list = []
    for c in range(num_classes):
        score_list.append(func(g[:, c], p[:, c]))

    return score_list

class loss_func(nn.Module):
    pass

