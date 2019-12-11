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
from utils.configs import parameters_grid, metric_list
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='training for MPM image classification')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data/', type=str, help='Path of data')
parser.add_argument('--img_path', default='data/MPM/', type=str, help='Path of data')
parser.add_argument('--gt_path', default='data/TMA_MPM_Summary_20191122_excluded_repeated.csv',
                    type=str, help='File of the groundtruth')
# parser.add_argument('--lr', '--learning_rate', default=1e-7, type=float, help='learning rate')
# parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, help='weight decay (like regularization)')
parser.add_argument('--n_batch', default=1, type=int, help='weight decay (like regularization)')

using_gpu = torch.cuda.is_available()
print("Using GPU: ", using_gpu)

gpu_count = torch.cuda.device_count()
print("Avaliable GPU:", gpu_count)

# Device configuration
device = torch.device('cuda' if using_gpu else 'cpu')
print("Using device: ", device)

args = parser.parse_args()
print(args)

n_jobs = gpu_count if using_gpu else 1
print("Parallel run with {} jobs tgt.".format(n_jobs))
# print("# Batch: ",)
input_tensor_size = (300, 300)


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
                                  # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ]

    val_input_transform_list = [cvtransforms.Resize(size=input_tensor_size, interpolation='BILINEAR'),
                                cvtransforms.ToTensor(),
                                # cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]

    train_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(train_input_transform_list)),
    ]

    base_dataset = mpImage_sorted_by_patient_dataset(img_dir=args.datapath,
                                                     multi_label_gt_path=gt_path,
                                                     transform=train_transforms[0])

    num_classes = base_dataset[0]["gt"].shape[-1]
    # Split data into cross-validation_set
    # cv_split_list = nfold_cross_validation(len(train_dataset), n_fold=2)
    # cv_split_list = nfold_cross_validation(4, n_fold=2)
    cv_split_list = leave_one_out_cross_validation(len(base_dataset))
    # cv_split_list = leave_one_out_cross_validation(2)

    running_states = ["train", "val"]
    n_fold = len(cv_split_list)

    # Grid Search
    parameters_grid["epochs"] = [args.epochs]
    parameters_grid["num_classes"] = [num_classes]
    parameters_grid["n_fold"] = [n_fold]
    parameters_grid["device"] = [device]

    list_parameters = ParameterGrid(parameters_grid)

    parametric_model_list = []
    for parameters in list_parameters:
        trainer_list = []
        specific_trainer = put_parameters_to_trainer_cv(**parameters)
        trainers_list = Parallel(n_jobs=n_jobs)(
            delayed(training_pipeline_per_fold)(nth_trainer=specific_trainer,
                                                  epochs=args.epochs,
                                                  nth_fold=nth_fold,
                                                  base_dataset_dict={
                                                      "base_dataset": mpImage_sorted_by_patient_dataset,
                                                      "datapath": args.datapath,
                                                      "gt_path": gt_path},
                                                  train_transform_list=train_input_transform_list,
                                                  val_transform_list=val_input_transform_list,
                                                  # train_data=train_dataset,
                                                  # val_data=val_dataset,
                                                  cv_splits=cv_split_list,
                                                  gpu_count=gpu_count,
                                                  n_batch=args.n_batch) for nth_fold in range(n_fold))

        specific_trainer = merge_all_fold_trainer(trainer_list)

    result_path = args.datapath + "patient_classify_result/"
    result_csv_name = result_path + 'result.csv'
    if os.path.exists(result_csv_name):
        out_df = pandas.read_csv(result_csv_name)
    else:
        out_df = base_dataset.multi_label_df.copy()
    label_list = base_dataset.label_name

    # label_name_list = train_val_dataset.label_name

    # label_list = ['Gleason score',"BCR", "AP", "EPE"]
    # label_list = ["BCR", "AP", "EPE"]

    for idx, label_name in enumerate(label_list):
        out_df = write_prediction_on_df_DL(trainers=parametric_model_list,
                                           df=out_df,
                                           state='val',
                                           patient_dataset=base_dataset,
                                           out_label_name=label_name,
                                           out_label_idx=idx
                                           )
        out_df = write_scores_on_df_DL(trainers=parametric_model_list,
                                       df=out_df,
                                       metrics=metric_list[1:],
                                       state='val',
                                       out_label=label_name,
                                       out_idx=idx)
        compare_model_cv(parametric_model_list, result_path,
                         output_label=label_name, output_idx=idx,
                         multi_label_classify=True, metrics=metric_list[1:],
                         )

        # some metric can't be evaluated when only one class is in the training set
        # compare_model_cv(parametric_model_list, result_path,
        #                  output_label=label_name, output_idx=idx,
        #                  multi_label_classify=True, metrics=metric_list[3], )

    compare_model_cv(parametric_model_list, result_path, metrics=['f1_by_sample'])
    out_df = write_scores_on_df_DL(trainers=parametric_model_list,
                                   df=out_df,
                                   metrics=metric_list[:1],
                                   state='val')
    out_df.fillna(' ')
    out_df.to_csv(result_path + 'result.csv', index=None, header=True)
