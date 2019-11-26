from utils.common_library import *
import argparse
from sklearn.model_selection import ParameterGrid
from utils.trainer import *
from utils.preprocess_data_loader import *
from utils.preprocess_data_transform import compose_input_output_transform
from cvtorchvision import cvtransforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import random
from utils.model import simple_transfer_classifier
from torch.autograd import Variable
import time
from utils.postprocessing_visualization import *
from decimal import Decimal
from utils.loss_metrics_evaluation import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='training for MPM image classification')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data/', type=str, help='Path of data')
parser.add_argument('--img_path', default='data/MPM/', type=str, help='Path of data')
parser.add_argument('--gt_path', default='data/TMA2_MPM_Summary_20191114.csv', type=str, help='File of the groundtruth')
# parser.add_argument('--lr', '--learning_rate', default=1e-7, type=float, help='learning rate')
# parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, help='weight decay (like regularization)')
parser.add_argument('--n_batch', default=1, type=int, help='weight decay (like regularization)')

using_gpu = torch.cuda.is_available()
print("Using GPU: ", using_gpu)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)



args = parser.parse_args()
print(args)

# print("# Batch: ",)
input_tensor_size = (800, 800)

def model_training_and_evaluate_testing(epochs,
                                        cross_val_indices,
                                        test_indices,
                                        trainer):
    pass



def parameters_dict_to_model_name(parameters_dict):
    pass

if __name__ == "__main__":
    img_path = args.img_path
    gt_path = args.gt_path


    # create dataset
    train_input_transform_list = [cvtransforms.Resize(size=input_tensor_size, interpolation='BILINEAR'),
                                 cvtransforms.RandomHorizontalFlip(),
                                 cvtransforms.RandomVerticalFlip(),
                                 cvtransforms.RandomRotation(90),
                                 cvtransforms.ToTensor(),
                                 cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    val_input_transform_list = [cvtransforms.Resize(size=input_tensor_size, interpolation='BILINEAR'),
                                  cvtransforms.ToTensor(),
                                  cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    train_transforms = [compose_input_output_transform(input_transform=cvtransforms.Compose(train_input_transform_list)),
                  ]

    train_dataset = torch.utils.data.ConcatDataset([
                mpImage_sorted_by_patient_dataset(img_dir=args.datapath,
                                                  multi_label_gt_path=gt_path,
                                                  transform=t) for t in train_transforms])

    val_transforms = [compose_input_output_transform(input_transform=cvtransforms.Compose(val_input_transform_list)),
                  ]

    val_dataset = torch.utils.data.ConcatDataset([
                mpImage_sorted_by_patient_dataset(img_dir=args.datapath,
                                                  multi_label_gt_path=gt_path,
                                                  transform=t) for t in val_transforms])

    num_classes = train_dataset[0]["gt"].shape[-1]
    # Split data into cross-validation_set
    cv_split_list = nfold_cross_validation(len(train_dataset), n_fold=5)

    running_states = ["train", "val"]
    n_fold = len(cv_split_list)


    cv_data_loaders = [{"train": DataLoader(dataset=train_dataset,
                                            batch_size=args.n_batch,
                                            sampler=SubsetRandomSampler(train_idx)),
                        "val": DataLoader(dataset=val_dataset,
                                          batch_size=args.n_batch,
                                          sampler=SubsetRandomSampler(val_idx))
                        } for train_idx, val_idx in cv_split_list]


    # Grid Search

    metric_list = ["f1_by_sample", "f1_by_label", "balanced_acc_by_label",
                   # "auc_by_label", "ap_by_label", "fmax_by_label"
                   ]
    parameters_grid = {"epochs": [args.epochs],
                       "num_classes": [num_classes],
                       "multi_label": [True],
                       "n_fold": [n_fold],
                       "performance_metrics_list" : [metric_list],
                       "device": [device],
                       # "p_model": ["resnext101_32x8d"],
                       "p_model": ["resnet18"],
                       "p_weight": [True],
                       "feat_ext": [True],
                       "lr":[1e-5],
                       "wd":[1e-2],
                       "input_res":[(3, input_tensor_size[0], input_tensor_size[1])],
                       "out_list": [False]
                       }
    list_parameters = ParameterGrid(parameters_grid)

    parametric_model_list = []
    for parameters in list_parameters:
        trainer_list = []
        specific_trainer = put_parameters_to_trainer(**parameters)
        for nth_fold in range(n_fold):
            print("{} {}th fold: {}".format("-" * 10, nth_fold, "-" * 10))
            specific_trainer.model_init()
            specific_trainer.model.to(device)
            running_loss = 0
            ran_data = 0
            for epoch in range(args.epochs):
                print("="*30)
                print("{} {}th epoch running: {}".format("="*10, epoch, "="*10))
                epoch_start_time = time.time()

                for running_state in running_states:
                    state_start_time = time.time()
                    for batch_idx, data in enumerate(cv_data_loaders[nth_fold][running_state]):
                        # print(batch_idx)
                        input = data['input']
                        gt = data['gt']
                        idx = data['idx']

                        input = Variable(input.view(-1, *(input.shape[2:]))).float().to(device)
                        gt = Variable(gt.view(-1, *(gt.shape[2:]))).float().to(device)
                        loss, predict = specific_trainer.running_model(input, gt, epoch=epoch,
                                                                       running_state=running_state, nth_fold=nth_fold, idx=idx)
                        ran_data += 1
                        running_loss += loss.item()
                    state_time_elapsed = time.time() - state_start_time
                    print("{}th epoch ({}) running time cost: {:.0f}m {:.0f}s".format(epoch, running_state, state_time_elapsed // 60,
                                                                                   state_time_elapsed % 60))
                    print('{}th epoch ({}) average loss: {}'.format(epoch, running_state, running_loss/ran_data))
                # print(loss)
                time_elapsed = time.time()-epoch_start_time

                print("{}{}th epoch running time cost: {:.0f}m {:.0f}s".format("-"*5, epoch, time_elapsed // 60, time_elapsed % 60))
            # specific_trainer.model = specific_trainer.model

        specific_trainer.evaluation()
        parametric_model_list.append(specific_trainer)

    result_path = args.datapath+"patient_classify_result/"
    # label_name_list = train_val_dataset.label_name
    compare_model_cv(parametric_model_list, result_path, metrics=['f1_by_sample'])
    # label_list = ['Gleason score',"BCR", "AP", "EPE"]
    label_list = ["BCR", "AP", "EPE"]

    for idx, label_name in enumerate(label_list):
        compare_model_cv(parametric_model_list, result_path,
                         output_label=label_name, output_idx=idx,
                         multi_label_classify=True, metrics=metric_list[1:3])

        # some metric can't be evaluated when only one class is in the training set
        compare_model_cv(parametric_model_list, result_path,
                         output_label=label_name, output_idx=idx,
                         multi_label_classify=True, metrics=metric_list[3], )







