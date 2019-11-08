from utils.common_library import *

class trainer(object):
    def __init__(self,
                 model,
                 model_name,
                 optimizer,
                 lr_scheduler_list,
                 n_batches,
                 loss_function,
                 performance_metrics,
                 locked_pretrained=True):
        self.model = model
        self.model_name = model_name
        self.model.weight_init(locked_pretrained=locked_pretrained)
        self.optimizer = optimizer
        self.lr_scheduler_list = lr_scheduler_list
        self.n_batches = n_batches
        self.loss_function = loss_function
        self.loss_stat = {"train":[],
                          "val":[],
                          "test":[]}
        self.best_model = None
        self.performance_metrics = performance_metrics
        self.performance_stat = {"train":[],
                          "val":[],
                          "test":[]}
        self.prediction_list = {"train":[],
                          "val":[],
                          "test":[]}
        self.gt_list = {"train":[],
                          "val":[],
                          "test":[]}



    def running_model(self, input, gt, running_status = "train"):
        predict = self.model(input)

        self.prediction_list[running_status].append(predict)
        self.gt_list[running_status].append(gt)

        loss = self.loss_function(predict, gt)

        self.loss_stat[running_status].append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, predict

    def evaluation(self, running_status):
        self.performance_stat[running_status].append(self.performance_metrics.eval(self.prediction_list, self.gt_list))
