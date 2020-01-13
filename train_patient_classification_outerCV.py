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
import importlib
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
import os
import subprocess
import pickle
from os import remove, system, listdir

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
parser.add_argument('--params_grid_config', default='params_3C', type=str, help='Path of params grid gonna be import')
parser.add_argument('--hpc', default=True, type=bool, help='Using hpc?')
parser.add_argument('--built_innerCV?', default=False, type=bool, help='Finished innerCV?')


args = parser.parse_args()
print(args)

def pick_optimal_params(score_df_path, num_params, picking_acc):
    try:
        score_df = pandas.read_csv(score_df_path)
    except FileNotFoundError:
        return False, 0
    row = score_df.shape[0]
    if row == num_params:
        max_idx_of_each_acc = score_df.idxmax(axis=0)
        max_idx_picking_acc = max_idx_of_each_acc[picking_acc]
        return True, max_idx_picking_acc
    else:
        return False, 0

def check_shd_run_outer(label_idx, label_name, num_params, all_inner_finish, params_picked):
    if label_idx == 0:
        picking_acc = "auc_by_img"
    else:
        picking_acc = "auc_by_patient"
    # while False in all_inner_finish[idx]:
    for nth_outer_fold in range(len(cv_split_list)):
        result_str = '{}th_fold_outerCV_'.format(nth_outer_fold)
        score_df_path = "{}{}_{}scores.csv".format(result_path,
                                         label_name,
                                         result_str)
        # score_df = pandas.read_csv(score_df_path)

        finished_inner, optimal_params_idx = pick_optimal_params(score_df_path=score_df_path,
                                                                 num_params=num_params,
                                                                 picking_acc=picking_acc,
                                                                 )
        params_picked[label_idx, nth_outer_fold] = optimal_params_idx
        all_inner_finish[label_idx, nth_outer_fold] = finished_inner

    return all_inner_finish, params_picked

if __name__ == "__main__":
    img_path = args.img_path
    gt_path = args.gt_path
    # parallel_running = args.parallel
    number_of_channels = args.input_C
    config_path = 'config/'

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
    len_dataset = len(base_dataset)
    outer_cv_train_idx = np.arange(len_dataset)
    # outer_cv_train_idx = np.load(args.outer_train_idx_path)
    cv_split_list = leave_one_patient_out_cross_validation(outer_cv_train_idx,
                                                           patient_deid=base_dataset.patient_deid_list[outer_cv_train_idx])
    # cv_split_list = leave_one_out_cross_validation(2)

    running_states = ["train", "val"]
    n_fold = len(cv_split_list)

    utils_module = __import__('config.' + args.params_grid_config)
    parameters_grid = getattr(utils_module, args.params_grid_config).parameters_grid
    list_parameters = ParameterGrid(parameters_grid)
    num_params = len(list_parameters)

    label_list = base_dataset.label_name



    metrics = img_metric_list
    # idx = args.predicting_label
    # label_name = label_list[idx]
    base_job_str = ['#!/bin/bash',
                    '#BSUB -J patient_classify # Job name',
                    '#BSUB -P acc_pandeg01a # allocation account',
                    '#BSUB -q gpu # queue'
                    '#BSUB -n 1 # number of compute cores',
                    '#BSUB -W 24:00 # walltime in HH:MM',
                    '#BSUB -R rusage[mem=8000] # 8 GB of memory requested',
                    '#BSUB -o pat_%J.stdout # output log (%J : JobID)',
                    '#BSUB -eo pat_%J.stderr # error log',
                    '#BSUB -L /bin/bash # Initialize the execution environment',
                    'module purge',
                    'module load anaconda3',
                    'module load cuda',
                    'source activate pytorchGPU',
                    ]
    params_performance = np.zeros((num_params, len(cv_split_list)))

    # params_df = pandas.DataFrame()
    params_path_npy = []
    for params_idx, parameters in enumerate(list_parameters):
        specific_trainer = put_parameters_to_trainer_cv(**parameters)
        params_fname = 'config/' + specific_trainer.model_name + '.params'
        pickle.dump(parameters, open(params_fname, 'wb'))
        params_path_npy.append(params_fname)
    params_path_npy_path = 'config/params_path.npy'
    np.save(params_path_npy_path, np.array(params_path_npy))

    for label_idx, label_name in enumerate(label_list):
        for nth_outer_fold, (train_idx, val_idx) in enumerate(cv_split_list):
            train_idx_npy = '{}{}th_fold_train_idx.npy'.format(config_path, nth_outer_fold)
            np.save(train_idx_npy,train_idx)
            for params_idx, parameters in enumerate(list_parameters):
                params_idx_path = '{}params_{}.npy'.format(config_path, params_idx)
                np.save(params_idx_path, np.array([params_idx]))
                specific_trainer = put_parameters_to_trainer_cv(**parameters)
                params_fname = config_path + specific_trainer.model_name + '.params'
                pickle.dump(parameters, open(params_fname, 'wb'))

                lsf_f_name = 'temp_submit_gpu_job_innerCV_{}.lsf'.format(nth_outer_fold)
                fn = open(lsf_f_name, 'w')
                base_py_cmd = 'python train_patient_classification_simplest_cv.py'

                base_py_cmd += ' --input_C='+str(number_of_channels)
                base_py_cmd += ' --params_path='+str(params_fname)

                base_py_cmd += ' --nth_fold='+str(nth_outer_fold)
                base_py_cmd += ' --train_idx_path='+train_idx_npy
                base_py_cmd += ' --label_predicting='+label_name
                base_py_cmd += ' --label_idx=' + str(label_idx)
                base_py_cmd += ' --params_npy='+params_path_npy_path
                base_py_cmd += ' --params_picked_idx_npy='+params_idx_path
                temp_job_str = base_job_str.copy()
                temp_job_str.append(base_py_cmd)
                for line in temp_job_str:
                    fn.write(line+'\n')
                fn.close()
                system('bsub < ' + lsf_f_name)
                # system('rm ' + lsf_f_name)

    all_inner_finish = np.zeros((len(label_list), n_fold)).astype(bool)
    params_picked = np.zeros((len(label_list), n_fold))

    # while False in all_inner_finish:


    while False in all_inner_finish:
        for label_idx, label_name in enumerate(label_list):
            all_inner_finish, params_picked = check_shd_run_outer(label_idx=label_idx,
                                                   label_name=label_name,
                                                   num_params=num_params,
                                                   all_inner_finish=all_inner_finish,
                                                   params_picked=params_picked
                                                   )
            if False not in all_inner_finish:
                # TODO
                train_idx_npy = 'outerCV_train_idx.npy'
                np.save(train_idx_npy, outer_cv_train_idx)
                params_idx_path = 'config/params_outerCV.npy'
                np.save(params_idx_path, params_picked)
                # specific_trainer = put_parameters_to_trainer_cv(**parameters)
                # params_fname = 'config/' + specific_trainer.model_name + '.params'
                # pickle.dump(parameters, open(params_fname, 'wb'))

                lsf_f_name = 'temp_submit_gpu_job_outerCV.lsf'
                fn = open(lsf_f_name, 'w')
                base_py_cmd = 'python train_patient_classification_simplest_cv.py'

                base_py_cmd += ' --input_C=' + str(number_of_channels)
                base_py_cmd += ' --params_path=outerCV'

                base_py_cmd += ' --nth_fold=' + str(n_fold)
                base_py_cmd += ' --train_idx_path=' + train_idx_npy
                base_py_cmd += ' --label_predicting=' + label_name
                base_py_cmd += ' --label_idx=' + str(label_idx)
                base_py_cmd += ' --params_npy=' + params_path_npy_path
                base_py_cmd += ' --params_picked_idx_npy=' + params_idx_path
                temp_job_str = base_job_str.copy()
                temp_job_str.append(base_py_cmd)
                for line in temp_job_str:
                    fn.write(line + '\n')
                fn.close()
                system('bsub < ' + lsf_f_name)
                # system('rm ' + lsf_f_name)

    print('Nested CV ended.')

















