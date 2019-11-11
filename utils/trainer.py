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
                 total_epochs,
                 use_pretrain_weight=True):
        self.model = model
        self.model_name = model_name
        if use_pretrain_weight is not True:
            self.weight_init(self.model)

        self.optimizer = optimizer
        self.lr_scheduler_list = lr_scheduler_list
        self.n_batches = n_batches
        self.loss_function = loss_function
        self.total_epochs = total_epochs
        self.loss_stat = {"train":[[] for i in range(self.total_epochs)],
                          "val":[[] for i in range(self.total_epochs)],
                          "test":[[] for i in range(self.total_epochs)]}
        self.best_model = None
        self.performance_metrics = performance_metrics
        self.performance_stat = {"train":[],
                          "val":[],
                          "test":[]}
        self.prediction_list = {"train":[[] for i in range(self.total_epochs)],
                          "val":[[] for i in range(self.total_epochs)],
                          "test":[[] for i in range(self.total_epochs)]}
        self.gt_list = {"train":[[] for i in range(self.total_epochs)],
                          "val":[[] for i in range(self.total_epochs)],
                          "test":[[] for i in range(self.total_epochs)]}
        self.old_epochs = 0


    def running_model(self, input, gt, epoch, running_state):
        predict = self.model(input)

        self.prediction_list[running_state][epoch].append(predict)
        self.gt_list[running_state][epoch].append(gt)

        loss = self.loss_function(predict, gt)

        self.loss_stat[running_state][epoch].append(loss)
        if running_state == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, predict

    def evaluation(self, running_state, epoch):
        metrics_dict = self.performance_metrics.eval(self.prediction_list[running_state][epoch],
                                          self.gt_list[running_state][epoch])
        print("# {} epoch performance:".format(epoch))
        for key, value in metrics_dict:
            print("{}:{}".format(key, value))
        self.performance_stat[running_state].append(metrics_dict)

    def inference(self, input):
        return self.model(input)

    def weight_init(self, *models, pretrained_weights):
        torch.manual_seed(0)
        for model in models:
            # print(model)
            modules_list = list(model.modules())

            for idx, module in enumerate(modules_list):
                # print('test',idx,  module)
                # print('test_idx', list(model.modules())[idx])
                if isinstance(module, nn.Conv2d):
                    """Searching next activation function"""
                    activation_not_found = True
                    alpha = 0
                    non_linearity = ''
                    next_idx = idx + 1
                    while activation_not_found:
                        # print(modules_list[next_idx])
                        if next_idx == len(modules_list):
                            alpha = 1
                            non_linearity = 'sigmoid'
                            activation_not_found = False
                        elif isinstance(modules_list[next_idx], nn.PReLU):
                            alpha = float(modules_list[next_idx].weight.mean())
                            non_linearity = 'leaky_relu'
                            activation_not_found = False
                        elif isinstance(modules_list[next_idx], nn.Sigmoid):
                            alpha = 1
                            non_linearity = 'sigmoid'
                            activation_not_found = False
                        elif isinstance(modules_list[next_idx], nn.ReLU):
                            alpha = 0
                            non_linearity = 'relu'
                            activation_not_found = False
                        next_idx += 1

                    if module.bias is None:
                        # print(module)
                        module.weight.data.zero_()
                        # module.bias.data.zero_()
                    else:
                        module.bias.data.zero_()
                        # makeDeltaOrthogonal(module.weight, nn.init.calculate_gain(non_linearity, alpha))
                        nn.init.kaiming_normal_(module.weight, a=alpha, nonlinearity=non_linearity)
                        # nn.init.xavier_normal_(module.weight, gain=math.sqrt(2/(1+0.25**2)))
                elif isinstance(module, nn.Linear):
                    # makeDeltaOrthogonal(module.weight, 1)
                    nn.init.orthogonal_(module.weight)
                    # if module.bias is not None:
                    #     module.bias.data.zero_()
                    # nn.init.kaiming_normal_(module.bias)
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
                    # print(module.weight, module.bias)
                    nn.init.uniform_(module.weight)
                    module.bias.data.zero_()
