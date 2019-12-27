from utils.common_library import *
import argparse
from sklearn.model_selection import ParameterGrid
from utils.trainer import *
from utils.preprocess_data_loader import *
from utils.preprocess_data_transform import compose_input_output_transform
from cvtorchvision import cvtransforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas
import random
from utils.model import simple_transfer_classifier
from torch.autograd import Variable
import time
from utils.postprocessing_visualization import *
from decimal import Decimal
from utils.loss_metrics_evaluation import *
from utils.configs_3C import *
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='training for MPM image classification')
parser.add_argument('--epochs', default=2, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data/', type=str, help='Path of data')
parser.add_argument('--img_path', default='data/MPM/', type=str, help='Path of data')
parser.add_argument('--gt_path', default='data/TMA_MPM.csv',
                    type=str, help='File of the groundtruth')
# parser.add_argument('--lr', '--learning_rate', default=1e-7, type=float, help='learning rate')
# parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, help='weight decay (like regularization)')
parser.add_argument('--n_batch', default=1, type=int, help='weight decay (like regularization)')
parser.add_argument('--predicting_label', default=0, type=int, help='label gonna be predicted')

using_gpu = torch.cuda.is_available()
print("Using GPU: ", using_gpu)

gpu_count = torch.cuda.device_count()
print("Avaliable GPU:", gpu_count)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

args = parser.parse_args()
print(args)

# print("# Batch: ",)
input_dim = parameters_grid["input_dim"][0]
input_tensor_res = (input_dim[-2], input_dim[-1])


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
    train_input_transform_list = [cvtransforms.Resize(size=input_tensor_res, interpolation='BILINEAR'),
                                  cvtransforms.RandomHorizontalFlip(),
                                  cvtransforms.RandomVerticalFlip(),
                                  cvtransforms.RandomRotation(90),
                                  cvtransforms.ToTensor(),
                                  # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ]

    train_input_transform_list = [cvtransforms.ToTensor()]

    val_input_transform_list = [cvtransforms.Resize(size=input_tensor_res, interpolation='BILINEAR'),
                                cvtransforms.ToTensor(),
                                # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]

    train_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(train_input_transform_list)),
        ]

    base_dataset = mpImage_4C_sorted_by_patient_dataset(img_dir=args.datapath,
                                                     multi_label_gt_path=gt_path,
                                                     transform=train_transforms[0])



    num_classes = base_dataset[0]["gt"].shape[-1]
    # Split data into cross-validation_set
    # cv_split_list = nfold_cross_validation(len(train_dataset), n_fold=2)
    # cv_split_list = nfold_cross_validation(4, n_fold=2)
    # cv_split_list = leave_one_out_cross_validation(len(base_dataset))
    cv_split_list = leave_one_patient_out_cross_validation(len(base_dataset),
                                                           patient_deid=base_dataset.patient_deid_list)
    # cv_split_list = leave_one_out_cross_validation(2)

    running_states = ["train", "val"]
    n_fold = len(cv_split_list)


    # Grid Search
    parameters_grid["epochs"] = [args.epochs]
    # parameters_grid["num_classes"] = [num_classes]
    parameters_grid["num_classes"] = [1]
    parameters_grid["n_fold"] = [n_fold]
    parameters_grid["device"] = [device]
    parameters_grid["n_batch"] = [args.n_batch]

    list_parameters = ParameterGrid(parameters_grid)

    label_list = base_dataset.label_name



    metrics = img_metric_list
    # idx = args.predicting_label
    # label_name = label_list[idx]

    result_path = args.datapath + "patient_classify_result/"
    result_csv_name = result_path + 'result.csv'
    if os.path.exists(result_csv_name):
        out_df = pandas.read_csv(result_csv_name)
    else:
        out_df = base_dataset.multi_label_df.copy

    # train_data = torch.utils.data.ConcatDataset([])

    loader = DataLoader(dataset=base_dataset, batch_size=args.n_batch, num_workers=0)

    start_time = time.time()
    for idx, data in enumerate(loader):
        input = data['input']
        gt = data['gt']
        deid = data['deid']
        row_idx = data['row_idx']

    print("4C Images loading time:{}".format(time.time()-start_time))

    base_dataset = mpImage_sorted_by_patient_dataset_2(img_dir=args.datapath,
                                                     multi_label_gt_path=gt_path,
                                                     transform=train_transforms[0])



    num_classes = base_dataset[0]["gt"].shape[-1]
    # Split data into cross-validation_set
    # cv_split_list = nfold_cross_validation(len(train_dataset), n_fold=2)
    # cv_split_list = nfold_cross_validation(4, n_fold=2)
    # cv_split_list = leave_one_out_cross_validation(len(base_dataset))
    cv_split_list = leave_one_patient_out_cross_validation(len(base_dataset),
                                                           patient_deid=base_dataset.patient_deid_list)
    # cv_split_list = leave_one_out_cross_validation(2)

    running_states = ["train", "val"]
    n_fold = len(cv_split_list)


    # Grid Search
    parameters_grid["epochs"] = [args.epochs]
    # parameters_grid["num_classes"] = [num_classes]
    parameters_grid["num_classes"] = [1]
    parameters_grid["n_fold"] = [n_fold]
    parameters_grid["device"] = [device]
    parameters_grid["n_batch"] = [args.n_batch]

    list_parameters = ParameterGrid(parameters_grid)

    label_list = base_dataset.label_name



    metrics = img_metric_list
    # idx = args.predicting_label
    # label_name = label_list[idx]

    result_path = args.datapath + "patient_classify_result/"
    result_csv_name = result_path + 'result.csv'
    if os.path.exists(result_csv_name):
        out_df = pandas.read_csv(result_csv_name)
    else:
        out_df = base_dataset.multi_label_df.copy

    # train_data = torch.utils.data.ConcatDataset([])

    loader = DataLoader(dataset=base_dataset, batch_size=args.n_batch, num_workers=0)

    start_time = time.time()
    for idx, data in enumerate(loader):
        input = data['input']
        gt = data['gt']
        deid = data['deid']
        row_idx = data['row_idx']

    print("RGB Images loading time:{}".format(time.time()-start_time))