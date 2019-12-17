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

    def eval(self, predict, gt, deid_list, states):
        performance_dict = {}
        prediction_list_by_state = {}
        gt_list_by_state = {}


        for metric in self.metrics_list:
            metric_func = globals()[metric]
            performance_dict[metric] = {}
            for state in states:
                performance_dict[metric][state] = []
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
                        deid = np.concatenate([deid_list[nth_fold][state][e] for nth_fold in range(self.nfold)], axis=0)
                        if "patient" in metric:
                            metric_scores = metric_func(g, p, deid=deid)
                        else:
                            metric_scores = metric_func(g, p)

                    performance_dict[metric][state].append(metric_scores)
                performance_dict[metric][state] = np.array(performance_dict[metric][state])

        return performance_dict


class performance_val_evaluater(object):
    def __init__(self, multi_label=True, metrics_list=["auc", "f_max", "ap"],
                 ):
        self.metrics_list = metrics_list
        self.multi_label = multi_label

    def eval(self, predict, gt):
        performance_dict = {}

        for metric in self.metrics_list:
            metric_func = globals()[metric]
            metric_score = metric_func(gt, predict)
            performance_dict[metric] = metric_score

        return performance_dict

def choose_best_params(performance_list, metric_list, choose_score_by=[]):
    performance_score_per_model = []
    vote_list = np.zeros((len(metric_list), len(performance_list)))
    for idx, metric in enumerate(metric_list):
        performance_of_specific_metric = np.array([performance[metric] for performance in performance_list])



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
        return -1*self.alpha*(torch.mean((1-predict)**self.gamma)*gt*torch.log(predict))

class multi_label_loss(nn.Module):
    def __init__(self, loss_function='BCE',
                 alpha=0.5,
                 gamma=2):
        super(multi_label_loss, self).__init__()
        self.alpha = torch.Tensor([alpha]).squeeze(-1)
        self.gamma = torch.Tensor([gamma]).squeeze(-1)
        self.loss_function = loss_function

    def forward(self, predict, gt):
        weight = torch.ones_like(predict)
        if predict.shape != gt.shape:
            gt = gt.unsqueeze(-1).repeat(1, 1, predict.shape[-1])
            weight[...,:-1] = weight[...,:-1]/(weight.shape[-1]-1)
        if self.loss_function == 'BCE':

            return nn.functional.binary_cross_entropy(predict, gt)
        elif self.loss_function == 'FL':
            self.alpha = self.alpha.to(predict.device)
            self.gamma = self.gamma.to(predict.device)
            fl = self.focal_loss(predict, gt) + self.focal_loss(1-predict, 1-gt)
            # print(fl)
            return fl.squeeze()

    def focal_loss(self, predict, gt):
        return -1*torch.mean(self.alpha*((1-predict)**self.gamma)*gt*torch.log(predict))

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

def r_max(gt, predict):
    recall, precision, threshold = metrics.precision_recall_curve(probas_pred=predict,
                                                             y_true=gt)
    # print("recall: ", recall)
    # print("precision: ", precision)

    recall = np.clip(recall, a_min=eps, a_max=1)
    precision = np.clip(precision, a_min=eps, a_max=1)

    f_score = 2*(recall*precision)/(recall+precision)
    # f_max = np.nanmax(f_score)
    r_max = recall[np.nanargmax(f_score)]
    return r_max

def p_max(gt, predict):
    recall, precision, threshold = metrics.precision_recall_curve(probas_pred=predict,
                                                             y_true=gt)
    # print("recall: ", recall)
    # print("precision: ", precision)

    recall = np.clip(recall, a_min=eps, a_max=1)
    precision = np.clip(precision, a_min=eps, a_max=1)

    f_score = 2*(recall*precision)/(recall+precision)
    # f_max = np.nanmax(f_score)
    p_max = precision[np.nanargmax(f_score)]
    return p_max

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

def balanced_acc_by_img(predict, gt):
    p = np.ones_like(predict)
    p[predict <= 0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    return metrics.balanced_accuracy_score(g, p)

def balanced_acc_by_patient(predict, gt):
    imgs_per_patient = 5
    gt = np.mean(gt.reshape(imgs_per_patient, -1, gt.shape[-1], order='F'), axis=0)
    predict = np.mean(predict.reshape(imgs_per_patient, -1, predict.shape[-1], order='F'), axis=0)
    p = np.ones_like(predict)
    p[predict <= 0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    return metrics.balanced_accuracy_score(g, p)

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

def f1_by_img(gt, predict):
    p = np.ones_like(predict)
    p[predict <= 0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    return metrics.f1_score(g, p)

def f1_by_patient(gt, predict):
    imgs_per_patient = 5
    gt = np.mean(gt.reshape(imgs_per_patient, -1, gt.shape[-1], order='F'), axis=0)
    predict = np.mean(predict.reshape(imgs_per_patient, -1, predict.shape[-1], order='F'), axis=0)
    p = np.ones_like(predict)
    p[predict <= 0.5] = 0
    p = p.astype(int)
    g = gt.astype(int)
    return metrics.f1_score(g, p)

def auc_by_img(gt, predict): return metrics.roc_auc_score(gt, predict)

def ap_by_img(gt, predict): return metrics.average_precision_score(gt, predict)

def fmax_by_img(gt, predict): return f_max(gt, predict)

def pmax_by_img(gt, predict): return p_max(gt, predict)

def rmax_by_img(gt, predict): return r_max(gt, predict)


def auc_by_patient(gt, predict, deid):
    return evaluate_by_patients(gt, predict, metrics.roc_auc_score, deid)

def ap_by_patient(gt, predict, deid):
    return evaluate_by_patients(gt, predict, metrics.average_precision_score, deid)

def fmax_by_patient(gt, predict, deid):
    return evaluate_by_patients(gt, predict, f_max, deid)

def pmax_by_patient(gt, predict, deid):
    return evaluate_by_patients(gt, predict, p_max, deid)

def rmax_by_patient(gt, predict, deid):
    return evaluate_by_patients(gt, predict, r_max, deid)



def auc_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, metrics.roc_auc_score)

def ap_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, metrics.average_precision_score)

def fmax_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, f_max)

def pmax_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, p_max)

def rmax_by_label(gt, predict):
    return evaluate_with_multi_label_classification(gt, predict, r_max)

def evaluate_with_multi_label_classification(g, p, func):

    num_classes = p.shape[-1]
    score_list = []
    for c in range(num_classes):
        score_list.append(func(g[:, c], p[:, c]))

    return score_list

def evaluate_by_patients(g, p, func, deid):
    unique_ids = np.unique(deid)
    pat_p = []
    pat_g = []
    for unique_id in unique_ids:
        pat_g.append(np.mean(g[deid==unique_id], axis=0))
        pat_p.append(np.mean(p[deid==unique_id], axis=0))
    return func(pat_g, pat_p)


class loss_func(nn.Module):
    pass

