from utils.common_library import *
# from utils import model
from utils.loss_metrics_evaluation import performance_evaluation
from torch.utils.tensorboard import SummaryWriter
from utils.model import *
from decimal import Decimal
from utils.loss_metrics_evaluation import *

class trainer(object):
    def __init__(self,
                 model,
                 model_name,
                 optimizer,
                 # n_batches,
                 total_epochs,
                 lr_scheduler_list=[],
                 loss_function=nn.BCELoss(),
                 performance_metrics=performance_evaluation(),
                 use_pretrain_weight=True):
        self.model = model
        self.model_name = model_name
        if use_pretrain_weight is not True:
            self.weight_init(self.model)

        self.optimizer = optimizer
        self.lr_scheduler_list = lr_scheduler_list
        # self.n_batches = n_batches
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
        predict_for_result, predict_for_loss_function = self.model(input)
        # print('p for result', predict_for_result.shape)
        # print('p for loss', predict_for_loss_function.shape)
        loss = self.loss_function(predict_for_loss_function, gt)




        if running_state == "train":
            # print("{} loss:{}".format(running_state, loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss, predict, gt = loss.detach().cpu(), \
                            predict_for_result.detach().cpu(), \
                            gt.detach().cpu()
        self.loss_stat[running_state][epoch].append(loss)
        self.prediction_list[running_state][epoch].append(predict)
        self.gt_list[running_state][epoch].append(gt)

        return loss, predict

    def evaluation(self, running_state, epoch):
        print("{} running state: {} {}".format("*" * 5, running_state, "*" * 5))

        self.prediction_list[running_state][epoch] = torch.cat(self.prediction_list[running_state][epoch], dim=0)
        self.gt_list[running_state][epoch] = torch.cat(self.gt_list[running_state][epoch], dim=0)

        metrics_dict = self.performance_metrics.eval(self.prediction_list[running_state][epoch],
                                          self.gt_list[running_state][epoch])

        # print("# {} epoch performance ({}):".format(epoch, running_state))
        for key, value in metrics_dict.items():
            print("{}: {}".format(key, value))
        self.performance_stat[running_state].append(metrics_dict)

        return metrics_dict

    def inference(self, input, device):
        self.model = self.model.to(device)
        input = input.to(device)
        return self.model(input)

    # def model_change_device(self, device):

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


def put_parameters_to_trainer(epochs=50,
                              num_classes=1,
                              device=torch.device('cpu'),
                              p_model="resnext101_32x8d",
                              p_weight=True,
                              feat_ext=False,
                              lr=1e-7,
                              wd=1e-2,
                              input_res=(3, 300, 300),
                              out_list=True):

    exclude_name_list = ["num_classes", "device", "epochs"]

    show_model_list = {"p_model": True,
                       "p_weight": True,
                       "feat_ext": False,
                       "lr": True,
                       "wd": True,
                       "input_res": False,
                       "out_list": True
                       }

    model_name = "TL"

    for key, show in show_model_list.items():
        if show:
            value = locals()[key]
            if type(value) == bool:
                if value:
                    model_name += "_"+key
            else:
                if type(value) == str:
                    model_name += "_"+value
                elif type(value) == int or type(value) == float:
                    model_name += "_{}={:.0e}".format(key, Decimal(value))
                elif key == "input_res":
                    model_name += "_{}={}".format(key, value[1])

    print(model_name)
    model = simple_transfer_classifier(num_classes=num_classes,
                                       input_size=input_res,
                                       pretrained_model_name=p_model,
                                       pretrain_weight=p_weight,
                                       feature_extracting=feat_ext,
                                       multi_classifier=out_list
                                       ).to(device)
    new_trainer = trainer(model=model,
                            model_name=model_name,
                            optimizer=torch.optim.Adam(lr=lr,
                                                       weight_decay=wd,
                                                       params=model.parameters(),
                                                       # amsgrad=True
                                                       ),
                            total_epochs=epochs,
                            lr_scheduler_list=[],
                            loss_function=bcel_multi_output())
    return new_trainer
