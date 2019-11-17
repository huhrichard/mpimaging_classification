from utils.common_library import *
import torchvision.models as models
from utils.custom_module import _ResBlock
import importlib


class simple_transfer_classifier(nn.Module):
    def __init__(self, num_classes, input_size,
                 module_prefix=None, pretrained_model_name="resnet18",
                 pretrain_weight=True, feature_extracting=True, multi_label=False):
        super(simple_transfer_classifier, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        # self.modules_employing = modules_employing
        self.module_prefix = module_prefix
        self.pretrain_weight = pretrain_weight
        self.feature_extracting = feature_extracting
        self.num_classes = num_classes
        # self.pretrained_network, self.feature_dim = get_pretrained_net(pretrained_model_name,
        #                                              input_size,
        #                                              module_prefix,
        #                                              pretrain_weight,
        #                                              feature_extracting)

        self.pretrained_network, self.feature_dim, self.net_as_list = get_pretrained_net(pretrained_model_name,
                                                                                         input_size,
                                                                                         module_prefix,
                                                                                         pretrain_weight,
                                                                                         feature_extracting)
        # print(self.pretrained_network)
        # self.feature_dim = (0,0,0)
        print("pretrained_model:{}, its output shape: {}".format(pretrained_model_name, self.feature_dim))
        # if self.pretrained_network_list.type
        if self.net_as_list:
            self.simplest_linear_act = []
            for feature in self.feature_dim:
                self.simplest_linear_act.append(self.create_linear_act_module(feature, multi_label, num_classes))
        else:
            self.simplest_linear_act = self.create_linear_act_module(self.feature_dim, multi_label, num_classes)

        # self.resblock1 = _ResBlock(in_channels=self.feature_dim[1], out_channels=self.feature_dim[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, input):
        if self.net_as_list:
            out_list = []
            for idx, net in enumerate(self.pretrained_network):
                input = net(input)
                avg_pool = self.avgpool(input).flatten(start_dim=1)
                out_list.append(self.simplest_linear_act[idx](avg_pool).unsqueeze(-1))
            out = (torch.cat(out_list, dim=-1).median(dim=-1))[0]

        else:
            features = self.pretrained_network(input)
            features = self.avgpool(features)
            features_flat = features.flatten(start_dim=1)
            # print(features_flat.shape)
            # print(self.feature_dim)
            out = self.simplest_linear_act(features_flat)[0]
        return out

    # def weight_init(self, use_pretrained_weight):

    def create_linear_act_module(self, feature_dim, multi_label, num_classes):

        if num_classes == 1:
            return nn.Sequential(*[nn.Linear(feature_dim[1], num_classes),
                                            nn.Sigmoid()])
        else:
            if multi_label:
                act = nn.Sigmoid()
            else:
                act = nn.Softmax()
            return nn.Sequential(*[nn.Linear(feature_dim[1], num_classes),
                                    act])





def get_pretrained_net(model_name, input_size, module_prefix=None, pretrain_weight=True,
                       feature_extracting=True):
        if module_prefix is None:
            if model_name == "inception_v3":
                pretrained_net = getattr(models, model_name)(pretrained=pretrain_weight, aux_logits=False)
            else:
                pretrained_net = getattr(models, model_name)(pretrained=pretrain_weight)
        else:
            m = importlib.import_module("torchvision.models."+module_prefix)
            pretrained_net = getattr(m, model_name)(pretrained=pretrain_weight)

        set_parameter_requires_grad(pretrained_net, feature_extracting)
        test_input = torch.zeros(1,input_size[0], input_size[1], input_size[2])
        # TODO: get the only layers want from 'modules_employing',
        # print(model_name)
        if "res" in model_name and "fcn" not in model_name:
            # Resnet for classification
            # feature_extractor_list = nn.Sequential(*list(pretrained_net.children())[:-2])
            feature_extractor_list = list(pretrained_net.children())[:-2]

        elif "densenet" in model_name:
            """ Densenet
            """
            feature_extractor_list = list(pretrained_net.children())[0]

        elif model_name == "inception_v3":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            # feature_extractor_list = nn.Sequential(*list(pretrained_net.children())[:-1])
            feature_extractor_list = list(pretrained_net.children())[:-1]

        else:
            print("Invalid model name, exiting...")
            exit()
        # print(feature_extractor_list)
        test_output_shape = []
        list_or_not = type(feature_extractor_list) == list
        if list_or_not:
            for feature_extractor in feature_extractor_list:
                test_input = feature_extractor(test_input)
                test_output_shape.append(test_input.shape)
        else:
            test_output_shape = feature_extractor_list(test_input).shape
        # print(test_output)
        return feature_extractor_list, test_output_shape, list_or_not

# def get_only_conv(network):
#     *list(res50_model.children())
#     only_conv = nn.Sequential()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




