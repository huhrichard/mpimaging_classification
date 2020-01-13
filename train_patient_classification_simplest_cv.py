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
from utils.metric_list import *
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
import os
import subprocess
import pickle

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='training for MPM image classification')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data/', type=str, help='Path of data')
parser.add_argument('--img_path', default='data/MPM/', type=str, help='Path of data')
parser.add_argument('--gt_path', default='data/TMA_MPM.csv',
                    type=str, help='File of the groundtruth')
# parser.add_argument('--lr', '--learning_rate', default=1e-7, type=float, help='learning rate')
# parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, help='weight decay (like regularization)')
# parser.add_argument('--predicting_label', default=0, type=int, help='label gonna be predicted')
# parser.add_argument('--parallel', default=False, type=bool, help='Run with joblib parallelization?')
parser.add_argument('--input_C', default=3, type=int, help='RGB (3) or Raw (4)?')
parser.add_argument('--params_path', default='', type=str, help='Path to pickle model parameters')
# parser.add_argument('--params_idx', default=0, type=int, help='idx of model parameters')
parser.add_argument('--nth_fold', default=0, type=int, help='n-th fold of CV?')
parser.add_argument('--train_idx_path', default='', type=str, help='Path of training index')
parser.add_argument('--label_predicting', default='', type=str, help='Which label is gonna be predict?')
parser.add_argument('--label_idx', default=0, type=int, help='Index of the label predicting')
parser.add_argument('--innerCV', default=True, type=bool, help='Performing inner CV?')
parser.add_argument('--params_picked_idx_npy', default='', type=str, help='params_list_outer_train')
parser.add_argument('--params_npy', default='', type=str, help='params_list_outer_train')


using_gpu = torch.cuda.is_available()
print("Using GPU: ", using_gpu)

gpu_count = torch.cuda.device_count()
print("Avaliable GPU:", gpu_count)

print('Available cuda', os.environ['CUDA_VISIBLE_DEVICES'] )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_mem = torch.cuda.get_device_properties(device).total_memory
print("Using device:{}, memory:{}".format(device, gpu_mem))

"""Get the current gpu usage.

Returns
-------
usage: dict
    Keys are device ids as integers.
    Values are memory usage as integers in MB.
"""
result = subprocess.check_output(
    [
        'nvidia-smi'
    ], encoding='utf-8')
print(result)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device_idx = torch.cuda.current_device()
print("Using device: ", device)

# if using_gpu:
#     n_jobs = gpu_count
# else:
#     n_jobs = 2
n_jobs = 1
print("Parallel run with {} jobs tgt.".format(n_jobs))

args = parser.parse_args()
print(args)

# print("# Batch: ",)
# input_dim = parameters_grid["input_dim"][0]
# input_tensor_res = (input_dim[-2], input_dim[-1])


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
    # parallel_running = args.parallel
    number_of_channels = args.input_C

    if number_of_channels == 3:
        # from utils.configs_3C import *
        base_dataset_class = mpImage_sorted_by_patient_dataset_2
        img_path = 'data/MPM/'
        result_path = args.datapath + "patient_classify_result/"
    elif number_of_channels == 4:
        # from utils.configs_4C import *
        base_dataset_class = mpImage_4C_sorted_by_patient_dataset
        img_path = 'data/MPM4C_16bit'
        result_path = args.datapath + "patient_classify_result_4C/"



    # create dataset
    train_input_transform_list = [
                                  # cvtransforms.Resize(size=input_tensor_res, interpolation='BILINEAR'),
                                  # cvtransforms.RandomHorizontalFlip(),
                                  # cvtransforms.RandomVerticalFlip(),
                                  # cvtransforms.RandomRotation(90),
                                  cvtransforms.ToTensor(),
                                  # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ]

    val_input_transform_list = [
                                # cvtransforms.Resize(size=input_tensor_res, interpolation='BILINEAR'),
                                cvtransforms.ToTensor(),
                                # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]

    train_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(train_input_transform_list)),
        ]

    base_dataset = base_dataset_class(img_dir=args.datapath,
                                     multi_label_gt_path=gt_path,
                                     transform=train_transforms[0])



    num_classes = base_dataset[0]["gt"].shape[-1]
    # Split data into cross-validation_set
    # cv_split_list = nfold_cross_validation(len(train_dataset), n_fold=2)
    # cv_split_list = nfold_cross_validation(4, n_fold=2)
    # cv_split_list = leave_one_out_cross_validation(len(base_dataset))
    cv_train_idx = np.load(args.train_idx_path)
    cv_split_list = leave_one_patient_out_cross_validation(cv_train_idx,
                                                           patient_deid=base_dataset.patient_deid_list[cv_train_idx])
    # cv_split_list = leave_one_out_cross_validation(2)

    running_states = ["train", "val"]
    n_fold = len(cv_split_list)



    # if parallel_running:
    #     trainers_list = Parallel(n_jobs=n_jobs, verbose=10)(
    #         delayed(training_pipeline_per_fold)(nth_trainer=specific_trainer,
    #                                           epochs=parameters['epochs'],
    #                                           nth_fold=nth_fold,
    #                                           base_dataset_dict= {"base_dataset": base_dataset_class,
    #                                                         "datapath": args.datapath,
    #                                                         "gt_path": gt_path},
    #                                           train_transform_list=train_input_transform_list,
    #                                           val_transform_list=val_input_transform_list,
    #                                           label_idx=idx,
    #                                           cv_splits=cv_split_list,
    #                                           gpu_count=gpu_count,
    #                                           n_batch=parameters['n_batch']
    #                                                   ) for nth_fold in range(n_fold))
    #     print(trainers_list)
    #     specific_trainer = merge_all_fold_trainer(trainers_list)
    # else:
    trained_list = []
    params_picked = np.load(args.params_picked_idx_npy)
    params_npy = np.load(args.params_npy)
    for nth_fold in range(n_fold):
        if params_picked.shape[0] < n_fold:
            params_file = params_npy[params_picked[0]]
        else:
            params_file = params_npy[params_picked[nth_fold]]
        parameters = pickle.load(open(params_file), 'rb')
        parameters["epochs"] = args.epochs
        input_dim = parameters["input_dim"]
        parameters["input_dim"] = (args.input_C, input_dim[0], input_dim[1])
        # parameters_grid["num_classes"] = [num_classes]
        parameters["num_classes"] = 1
        parameters["n_fold"] = n_fold
        parameters["device"] = device

        # list_parameters = ParameterGrid(parameters_grid)

        label_list = base_dataset.label_name

        metrics = img_metric_list
        # idx = args.predicting_label
        # label_name = label_list[idx]

        idx = args.label_idx
        label_name = args.label_predicting
        # for idx, label_name in enumerate(label_list):

        print("Current label predicting:", label_name)
        if idx == 0:
            metrics = img_metric_list
        else:
            metrics = img_metric_list + patient_metric_list
        print(metrics)
        # parametric_model_list = []

        # for parameters in list_parameters:
        # trainer_list = []
        parameters['performance_metrics_list'] = metrics
        specific_trainer = put_parameters_to_trainer_cv(**parameters)
        trained_specific_trainer = training_pipeline_per_fold(nth_trainer=specific_trainer,
                                                      epochs=parameters['epochs'],
                                                      nth_fold=nth_fold,
                                                      base_dataset_dict= {"base_dataset": base_dataset_class,
                                                                    "datapath": args.datapath,
                                                                    "gt_path": gt_path},
                                                      train_transform_list=train_input_transform_list,
                                                      val_transform_list=val_input_transform_list,
                                                      label_idx=idx,
                                                      cv_splits=cv_split_list,
                                                      gpu_count=gpu_count,
                                                      n_batch=parameters['n_batch'],
                                                      # params_list=
                                                      )
        trained_list.append(trained_specific_trainer)

    merged_trainer = merge_all_fold_trainer(trained_list)
    merged_trainer.evaluation()
        # parametric_model_list.append(specific_trainer)

        # label_name_list = train_val_dataset.label_name

        # label_list = ['Gleason score',"BCR", "AP", "EPE"]
        # label_list = ["BCR", "AP", "EPE"]

    if args.innerCV:
        result_str = '{}th_fold_outerCV_'.format(args.nth_fold)
    else:
        result_str = ''

    score_csv_name = "{}{}_{}scores.csv".format(result_path,
                                             label_name,
                                             result_str)
    if os.path.exists(score_csv_name):
        score_df = pandas.read_csv(score_csv_name)
    else:
        score_df = pd.DataFrame()

    score_df = write_scores_on_df_DL_score_df(trainer=merged_trainer,
                                              params_path=args.params_path,
                                              df=score_df,
                                              metrics=metrics,
                                              state='val',
                                              out_label=label_name,
                                              params_idx=args.params_idx,
                                              nth_fold=args.nth_fold)

    score_df.to_csv(score_csv_name, index=None, header=True)

    if not args.innerCV:
        score_pred_csv_name = "{}result.csv".format(result_path)

        if os.path.exists(score_csv_name):
            score_pred_df = pandas.read_csv(score_csv_name)
        else:
            score_pred_df = base_dataset.multi_label_df.copy()

        score_pred_df = write_prediction_on_df_DL(trainer=merged_trainer,
                                           df=score_pred_df,
                                           state='val',
                                           patient_dataset=base_dataset,
                                           out_label_name=label_name,
                                           out_label_idx=0
                                           )

        # compare_model_cv(specific_trainer, result_path,
        #                  output_label=label_name, output_idx=0,
        #                  multi_label_classify=False, metrics=metrics,
        #                  )
        score_pred_df.fillna(' ')
        score_pred_df.to_csv(score_pred_csv_name, index=None, header=True)


    # score_df.fillna(' ')


            # some metric can't be evaluated when only one class is in the training set
            # compare_model_cv(parametric_model_list, result_path,
            #                  output_label=label_name, output_idx=idx,
            #                  multi_label_classify=True, metrics=metric_list[3], )


