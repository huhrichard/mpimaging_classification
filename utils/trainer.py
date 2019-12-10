from utils.common_library import *
# from utils import model
from utils.loss_metrics_evaluation import performance_evaluation
from utils.radam import *
from torch.utils.tensorboard import SummaryWriter
from utils.model import *
from decimal import Decimal
from utils.loss_metrics_evaluation import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
from torch.autograd import Variable
from cvtorchvision import cvtransforms
from utils.preprocess_data_transform import compose_input_output_transform


# class trainer(object):
#     def __init__(self,
#                  model_class,
#                  model_dict,
#                  model_name,
#                  optimizer,
#                  n_fold,
#                  total_epochs,
#                  lr_scheduler_list=[],
#                  loss_function=nn.BCELoss(),
#                  performance_metrics=performance_evaluation(),
#                  use_pretrain_weight=True):
#         self.model_class = model_class
#         self.model_dict = model_dict
#         self.model = self.model_class(**self.model_dict)
#         self.model_name = model_name
#         if use_pretrain_weight is not True:
#             self.weight_init(self.model)
#
#         self.optimizer = optimizer
#         self.lr_scheduler_list = lr_scheduler_list
#         # self.n_batches = n_batches
#         self.loss_function = loss_function
#         self.total_epochs = total_epochs
#         self.loss_stat = {"train": [[] for i in range(self.total_epochs)],
#                           "val": [[] for i in range(self.total_epochs)],
#                           "test": [[] for i in range(self.total_epochs)]}
#         self.best_model = None
#         self.performance_metrics = performance_metrics
#         self.performance_stat = {"train": [],
#                                  "val": [],
#                                  "test": []}
#         self.prediction_list = {"train": [[] for i in range(self.total_epochs)],
#                                 "val": [[] for i in range(self.total_epochs)],
#                                 "test": [[] for i in range(self.total_epochs)]}
#         self.gt_list = {"train": [[] for i in range(self.total_epochs)],
#                         "val": [[] for i in range(self.total_epochs)],
#                         "test": [[] for i in range(self.total_epochs)]}
#         self.old_epochs = 0
#
#     def running_model(self, input, gt, epoch, running_state):
#         predict_for_result, predict_for_loss_function = self.model(input)
#         # print('p for loss', predict_for_loss_function)
#         # print('p for result', predict_for_result)
#         # print('gt', gt)
#         loss = self.loss_function(predict_for_loss_function, gt)
#
#         if running_state == "train":
#             # print("{} loss:{}".format(running_state, loss))
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#         loss, predict, gt = loss.detach().cpu(), \
#                             predict_for_result.detach().cpu(), \
#                             gt.detach().cpu()
#
#         self.loss_stat[running_state][epoch].append(loss)
#         self.prediction_list[running_state][epoch].append(predict)
#         self.gt_list[running_state][epoch].append(gt)
#
#         return loss, predict
#
#     def evaluation(self, running_state, epoch):
#         print("{} running state: {} {}".format("*" * 5, running_state, "*" * 5))
#
#         self.prediction_list[running_state][epoch] = torch.cat(self.prediction_list[running_state][epoch], dim=0)
#         self.gt_list[running_state][epoch] = torch.cat(self.gt_list[running_state][epoch], dim=0)
#
#         metrics_dict = self.performance_metrics.eval(self.prediction_list[running_state][epoch],
#                                                      self.gt_list[running_state][epoch])
#
#         # print("# {} epoch performance ({}):".format(epoch, running_state))
#         self.performance_stat[running_state].append(metrics_dict)
#
#         return metrics_dict
#
#     def inference(self, input, device):
#         self.model = self.model.to(device)
#         input = input.to(device)
#         return self.model(input)
#
#     # def model_change_device(self, device):
#
#     def weight_init(self, *models, pretrained_weights):
#         torch.manual_seed(0)
#         for model in models:
#             # print(model)
#             modules_list = list(model.modules())
#
#             for idx, module in enumerate(modules_list):
#                 # print('test',idx,  module)
#                 # print('test_idx', list(model.modules())[idx])
#                 if isinstance(module, nn.Conv2d):
#                     """Searching next activation function"""
#                     activation_not_found = True
#                     alpha = 0
#                     non_linearity = ''
#                     next_idx = idx + 1
#                     while activation_not_found:
#                         # print(modules_list[next_idx])
#                         if next_idx == len(modules_list):
#                             alpha = 1
#                             non_linearity = 'sigmoid'
#                             activation_not_found = False
#                         elif isinstance(modules_list[next_idx], nn.PReLU):
#                             alpha = float(modules_list[next_idx].weight.mean())
#                             non_linearity = 'leaky_relu'
#                             activation_not_found = False
#                         elif isinstance(modules_list[next_idx], nn.Sigmoid):
#                             alpha = 1
#                             non_linearity = 'sigmoid'
#                             activation_not_found = False
#                         elif isinstance(modules_list[next_idx], nn.ReLU):
#                             alpha = 0
#                             non_linearity = 'relu'
#                             activation_not_found = False
#                         next_idx += 1
#
#                     if module.bias is None:
#                         # print(module)
#                         module.weight.data.zero_()
#                         # module.bias.data.zero_()
#                     else:
#                         module.bias.data.zero_()
#                         # makeDeltaOrthogonal(module.weight, nn.init.calculate_gain(non_linearity, alpha))
#                         nn.init.kaiming_normal_(module.weight, a=alpha, nonlinearity=non_linearity)
#                         # nn.init.xavier_normal_(module.weight, gain=math.sqrt(2/(1+0.25**2)))
#                 elif isinstance(module, nn.Linear):
#                     # makeDeltaOrthogonal(module.weight, 1)
#                     nn.init.orthogonal_(module.weight)
#                     # if module.bias is not None:
#                     #     module.bias.data.zero_()
#                     # nn.init.kaiming_normal_(module.bias)
#                 elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
#                     # print(module.weight, module.bias)
#                     nn.init.uniform_(module.weight)
#                     module.bias.data.zero_()


class cv_trainer(object):
    def __init__(self,
                 model_class,
                 model_dict,
                 model_name,
                 optimizer_dict,
                 n_fold,
                 total_epochs,
                 lr_scheduler_list=[],
                 loss_function=nn.BCELoss(),
                 performance_metrics=performance_evaluation_cv(),
                 use_pretrain_weight=True,
                 train_data_normal=False):
        self.model_class = model_class
        self.model_dict = model_dict
        self.model_name = model_name
        self.use_pretrain_weight = use_pretrain_weight
        self.optimizer_dict = optimizer_dict

        """Do not declare self.model in method"""
        # self.model = self.model_init()
        self.model_init()
        # print(optimizer)


        # self.optimizer =

        self.lr_scheduler_list = lr_scheduler_list
        # self.n_batches = n_batches
        self.n_fold = n_fold
        self.loss_function = loss_function
        self.total_epochs = total_epochs
        self.train_data_normal = train_data_normal

        self.loss_stat = [{"train": [[] for i in range(self.total_epochs)],
                           "val": [[] for i in range(self.total_epochs)],
                           "test": [[] for i in range(self.total_epochs)]} for n in range(n_fold)]
        self.best_model = None
        self.performance_metrics = performance_metrics
        self.performance_stat = [{"train": [],
                                  "val": [],
                                  "test": []}]
        self.prediction_list = [{"train": [[] for i in range(self.total_epochs)],
                                 "val": [[] for i in range(self.total_epochs)],
                                 "test": [[] for i in range(self.total_epochs)]} for n in range(n_fold)]
        self.gt_list = [{"train": [[] for i in range(self.total_epochs)],
                         "val": [[] for i in range(self.total_epochs)],
                         "test": [[] for i in range(self.total_epochs)]} for n in range(n_fold)]
        self.idx_list = [{"train": [[] for i in range(self.total_epochs)],
                         "val": [[] for i in range(self.total_epochs)],
                         "test": [[] for i in range(self.total_epochs)]} for n in range(n_fold)]
        self.old_epochs = 0

    def running_model(self, input, gt, epoch, running_state, nth_fold, idx):
        predict_for_result, predict_for_loss_function = self.model(input)
        # self.check_grad()
        # print('p for loss', predict_for_loss_function)
        # print('p for result', predict_for_result)
        # print('gt', gt)
        loss = self.loss_function(predict_for_loss_function, gt)
        # print(self.optimizer.param_groups)
        # print("{} loss:{}".format(running_state, loss))
        if running_state == "train":

            # a = list(self.model.parameters())[0].clone()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # b = list(self.model.parameters())[0].clone()
            # print('a=b?', torch.equal(a.data, b.data))


        # detach all to release gpu memory
        loss_detached, predict_detached, gt_detached = loss.detach().cpu(), \
                            predict_for_result.detach().cpu(), \
                            gt.detach().cpu()

        if running_state == "val":
            if epoch % 10 == 0:
                print("predict:", predict_detached)
                print("gt:", gt_detached)

        idx = idx.detach().cpu()

        self.loss_stat[nth_fold][running_state][epoch].append(loss_detached)
        self.prediction_list[nth_fold][running_state][epoch].append(predict_detached)
        self.gt_list[nth_fold][running_state][epoch].append(gt_detached)
        self.idx_list[nth_fold][running_state][epoch].append(idx)

        return loss_detached, predict_detached

    def evaluation(self):
        # print("{} running state: {} {}".format("*" * 5, running_state, "*" * 5))

        # running_states = ["train", "val"]
        running_states = ["val"]
        for nth_fold in range(self.n_fold):
            for running_state in running_states:
                np_p = [self.torch_tensor_np(torch.cat(self.prediction_list[nth_fold][running_state][epoch], dim=0)) for epoch in range(self.total_epochs)]

                np_t = [self.torch_tensor_np(torch.cat(self.gt_list[nth_fold][running_state][epoch], dim=0)) for epoch in range(self.total_epochs)]
                self.prediction_list[nth_fold][running_state] = np_p
                self.gt_list[nth_fold][running_state] = np_t
                # print(np_p, np_t)
        metrics_dict = self.performance_metrics.eval(self.prediction_list,
                                             self.gt_list, running_states)


        self.performance_stat = metrics_dict
        # print("# {} epoch performance ({}):".format(epoch, running_state))
        # self.performance_stat[running_state].append(metrics_dict)

        return metrics_dict

    def check_grad(self):
        for param in self.model.parameters():
            print(param.requires_grad)

    @staticmethod
    def torch_tensor_np(tensor):
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()

        np_array = tensor.detach().numpy()
        # if np_array.shape[-1] == 1:
        #     np_array = np.squeeze(np_array)
        return np_array

    def inference(self, input, device):
        self.model = self.model.to(device)
        input = input.to(device)
        return self.model(input)

    # def model_change_device(self, device):
    def model_init(self):
        self.model = self.model_class(**self.model_dict)

        if self.use_pretrain_weight is not True:
            self.weight_init(self.model)
        self.opti_init()


    def opti_init(self):
        self.optimizer = self.optimizer_dict['optim'](lr=self.optimizer_dict['lr'],
                                                      weight_decay=self.optimizer_dict['wd'],
                                                        params=self.model.parameters())

    # def model_init(self):
    #     self.model = self.model_class(**self.model_dict)
    #
    #     if self.use_pretrain_weight is not True:
    #         self.weight_init(self.model)

        # return model

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


def put_parameters_to_trainer_cv(epochs=50,
                                 multi_label=True,
                                 n_fold=10,
                                 performance_metrics_list=["auc", "f_max", "ap"],
                                 num_classes=1,
                                 device=torch.device('cpu'),
                                 p_model="resnext101_32x8d",
                                 p_weight=True,
                                 feat_ext=False,
                                 optim=torch.optim.Adam,
                                 lr=1e-7,
                                 wd=1e-2,
                                 input_res=(3, 300, 300),
                                 out_list=True,
                                 loss='BCE',
                                 train_data_normal=False):
    exclude_name_list = ["num_classes", "device", "epochs"]

    show_model_list = {"p_model": True,
                       "p_weight": True,
                       "feat_ext": True,
                       "lr": True,
                       "wd": True,
                       "input_res": False,
                       "out_list": True,
                       "loss": True,
                       "train_data_normal": True
                       }

    model_name = "TL"

    for key, show in show_model_list.items():
        if show:
            value = locals()[key]
            if type(value) == bool:
                if value:
                    model_name += "_" + key
            else:
                if type(value) == str:
                    model_name += "_" + value
                elif type(value) == int or type(value) == float:
                    model_name += "_{}={:.0e}".format(key, Decimal(value))
                elif key == "input_res":
                    model_name += "_{}={}".format(key, value[1])

    print(model_name)
    model_dict = {'num_classes': num_classes,
                  'input_size': input_res,
                  'pretrained_model_name': p_model,
                  'pretrain_weight': p_weight,
                  'feature_extracting': feat_ext,
                  'multi_classifier': out_list,
                  'multi_label': multi_label
                  }

    # model = simple_transfer_classifier(
    #                                    ).to(device)
    new_trainer = cv_trainer(model_class=simple_transfer_classifier,
                             model_dict=model_dict,
                             model_name=model_name,
                             optimizer_dict={'optim': torch.optim.Adam,
                                             # optimizer={'optim': RAdam,
                                        'lr': lr,
                                        'wd': wd},
                             n_fold=n_fold,
                             performance_metrics=performance_evaluation_cv(nfold=n_fold,
                                                                           multi_label=multi_label,
                                                                        metrics_list=performance_metrics_list,
                                                                           total_epochs=epochs),
                             total_epochs=epochs,
                             lr_scheduler_list=[],
                             loss_function=multi_label_loss(loss_function=loss),
                             train_data_normal=train_data_normal)

    return new_trainer

def merge_all_fold_trainer(list_of_trainer):
    first_trainer = list_of_trainer[0]
    for idx, nth_folder_trainer in enumerate(list_of_trainer):
        if idx != 0:
            first_trainer.loss_stat[idx] = nth_folder_trainer.loss_stat[idx]
            first_trainer.prediction_list[idx] = nth_folder_trainer.prediction_list[idx]
            first_trainer.gt_list[idx] = nth_folder_trainer.gt_list[idx]
            first_trainer.idx_list[idx] = nth_folder_trainer.idx_list[idx]
    return first_trainer

def training_pipeline_per_fold(nth_trainer, epochs, nth_fold, base_dataset_dict,
                               train_transform_list, val_transform_list,
                               cv_splits, gpu_count, n_batch):
    cv_split = cv_splits[nth_fold]
    train_transform_list_temp = train_transform_list.copy()
    val_transform_list_temp = val_transform_list.copy()

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(nth_fold % gpu_count))
        print(device)
    else:
        device = torch.device('cpu')
    if not nth_trainer.train_data_normal:
        train_normal = cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        train_mean, train_std = get_normalization_mean_std_from_training_set(base_dataset_dict=base_dataset_dict,
                                                                             train_idx=cv_split[0],
                                                                             device=device,
                                                                             train_transform_list=train_transform_list_temp,
                                                                             n_batch=n_batch)
        train_normal = cvtransforms.Normalize(train_mean, train_std)
    print(train_normal)
    train_transform_list_temp.append(train_normal)
    val_transform_list_temp.append(train_normal)

    train_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(train_transform_list_temp)),
        ]
    train_data = torch.utils.data.ConcatDataset([
        base_dataset_dict["base_dataset"](img_dir=base_dataset_dict["datapath"],
                                          multi_label_gt_path=base_dataset_dict["gt_path"],
                                          transform=t) for t in train_transforms])

    val_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(val_transform_list_temp)),
    ]
    val_data = torch.utils.data.ConcatDataset([
        base_dataset_dict["base_dataset"](img_dir=base_dataset_dict["datapath"],
                                          multi_label_gt_path=base_dataset_dict["gt_path"],
                                          transform=t) for t in val_transforms])

    train_data_loader = DataLoader(dataset=train_data, batch_size=n_batch, sampler=SubsetRandomSampler(cv_split[0]))
    val_data_loader = DataLoader(dataset=val_data, batch_size=n_batch, sampler=SubsetRandomSampler(cv_split[1]))
    print("{} {}th fold: {}".format("-" * 10, nth_fold, "-" * 10))
    nth_trainer.model_init()
    nth_trainer.model.to(device)
    running_loss = 0
    ran_data = 0
    running_states = ['train', 'val']
    for epoch in range(epochs):
        print("=" * 30)
        print("{} {}th epoch running: {}".format("=" * 10, epoch, "=" * 10))
        epoch_start_time = time.time()

        for running_state in running_states:
            state_start_time = time.time()
            if running_state == "train":
                cv_data_loader = train_data_loader
            else:
                cv_data_loader = val_data_loader
            for batch_idx, data in enumerate(cv_data_loader):
                # print(batch_idx)
                input = data['input']
                gt = data['gt']
                idx = data['idx']
                # print(idx, gt)
                input = Variable(input.view(-1, *(input.shape[2:]))).float().to(device)
                gt = Variable(gt.view(-1, *(gt.shape[2:]))).float().to(device)
                # print(gt)
                loss, predict = nth_trainer.running_model(input, gt, epoch=epoch,
                                                               running_state=running_state, nth_fold=nth_fold,
                                                               idx=idx)
                ran_data += 1
                running_loss += loss.item()

            state_time_elapsed = time.time() - state_start_time
            print("{}th epoch ({}) running time cost: {:.0f}m {:.0f}s".format(epoch, running_state,
                                                                              state_time_elapsed // 60,
                                                                              state_time_elapsed % 60))
            print('{}th epoch ({}) average loss: {}'.format(epoch, running_state, running_loss / ran_data))
            running_loss = 0
            ran_data = 0
        # print(loss)
        time_elapsed = time.time() - epoch_start_time

        print("{}{}th epoch running time cost: {:.0f}m {:.0f}s".format("-" * 5, epoch, time_elapsed // 60,
                                                                       time_elapsed % 60))

    nth_trainer.model = None
    return nth_trainer

def get_normalization_mean_std_from_training_set(base_dataset_dict, train_idx, device,
                                                 train_transform_list, n_batch):
    sampler = SubsetRandomSampler(train_idx)
    simpler_transform = [train_transform_list[0], train_transform_list[-1]]
    # print(simpler_transform)
    train_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(simpler_transform)),
        ]
    base_dataset = base_dataset_dict["base_dataset"](img_dir=base_dataset_dict["datapath"],
                                                     multi_label_gt_path=base_dataset_dict["gt_path"],
                                                     transform=train_transforms[0])
    train_data_loader = DataLoader(dataset=base_dataset, sampler=sampler)
    train_data_stack = []
    for batch_idx, data in enumerate(train_data_loader):
        input = data['input'].to(device)
        # if train_data_stack.shape[0] == 0:
        train_data_stack.append(input.transpose_(0,-3).flatten(start_dim=1))
    torch_stacked_input = torch.cat(train_data_stack, dim=1)
    train_mean = torch_stacked_input.mean(dim=1).to(torch.device('cpu'))
    train_std = torch_stacked_input.std(dim=1).to(torch.device('cpu'))
    return train_mean, train_std



def put_parameters_to_trainer_cv_nested(epochs=50,
                                 multi_label=True,
                                 n_fold=10,
                                 performance_metrics_list=["auc", "f_max", "ap"],
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
                    model_name += "_" + key
            else:
                if type(value) == str:
                    model_name += "_" + value
                elif type(value) == int or type(value) == float:
                    model_name += "_{}={:.0e}".format(key, Decimal(value))
                elif key == "input_res":
                    model_name += "_{}={}".format(key, value[1])

    print(model_name)
    model_dict = {'num_classes': num_classes,
                  'input_size': input_res,
                  'pretrained_model_name': p_model,
                  'pretrain_weight': p_weight,
                  'feature_extracting': feat_ext,
                  'multi_classifier': out_list,
                  'multi_label': multi_label
                  }

    # model = simple_transfer_classifier(
    #                                    ).to(device)
    new_trainer = cv_trainer(model_class=simple_transfer_classifier,
                             model_dict=model_dict,
                             model_name=model_name,
                             optimizer={'optim': torch.optim.Adam,
                                        'lr': lr,
                                        'wd': wd},
                             n_fold=n_fold,
                             performance_metrics=performance_evaluation_cv(nfold=n_fold,
                                                                           multi_label=multi_label,
                                                                        metrics_list=performance_metrics_list,
                                                                           total_epochs=epochs),
                             total_epochs=epochs,
                             lr_scheduler_list=[],
                             loss_function=bcel_multi_output(),
                             )

    return new_trainer
