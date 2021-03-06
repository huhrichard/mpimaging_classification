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
from config import con
from joblib import Parallel, delayed
import concurrent.futures
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

gpu_count = torch.cuda.device_count()
print("Avaliable GPU:", gpu_count)

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

def run_inner_cv(parameters,
                 inner_cv,
                 train_dataset,
                 val_dataset,
                 n_batch,
                 epochs,
                 iterate_idx,
                 gpu_count):
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(idx % gpu_count))

    parameters['device'] = device
    trainer_inner_cv = put_parameters_to_trainer_cv(**parameters)
    for inner_nth_fold in range(len(inner_cv)):
        print("{} outer {}th fold, inner {}th fold: {}".format("-" * 10, outer_nth_fold, inner_nth_fold, "-" * 10))
        inner_train_idx, inner_val_idx = inner_cv[inner_nth_fold]
        trainer_inner_cv.model_init()
        trainer_inner_cv.model.to(iterate_idx)
        running_loss = 0
        ran_data = 0
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=n_batch,
                                       sampler=SubsetRandomSampler(inner_train_idx))
        val_data_loader = DataLoader(dataset=val_dataset,
                                     batch_size=n_batch,
                                     sampler=SubsetRandomSampler(inner_val_idx))

        for epoch in range(epochs):
            print("=" * 30)
            print("{} inner {}th fold, {}th epoch running: {}".format("=" * 10, inner_nth_fold, epoch, "=" * 10))
            epoch_start_time = time.time()

            for running_state in inner_running_states:
                state_start_time = time.time()
                if running_state == "train":
                    data_loader = train_data_loader
                else:
                    data_loader = val_data_loader
                for batch_idx, data in enumerate(data_loader):
                    # print(batch_idx)
                    input = data['input']
                    gt = data['gt']
                    iterate_idx = data['idx']

                    input = Variable(input.view(-1, *(input.shape[2:]))).float().to(iterate_idx)
                    gt = Variable(gt.view(-1, *(gt.shape[2:]))).float().to(iterate_idx)
                    loss, predict = trainer_inner_cv.running_model(input, gt, epoch=epoch,
                                                                   running_state=running_state, nth_fold=outer_nth_fold,
                                                                   deid=iterate_idx)
                    ran_data += 1
                    running_loss += loss.item()
                state_time_elapsed = time.time() - state_start_time
                print("{}th epoch ({}) running time cost: {:.0f}m {:.0f}s".format(epoch, running_state,
                                                                                  state_time_elapsed // 60,
                                                                                  state_time_elapsed % 60))
                print('{}th epoch ({}) average loss: {}'.format(epoch, running_state, running_loss / ran_data))
            # print(loss)
            time_elapsed = time.time() - epoch_start_time

            print("{}{}th epoch running time cost: {:.0f}m {:.0f}s".format("-" * 5, epoch, time_elapsed // 60,
                                                                           time_elapsed % 60))
    trainer_inner_cv.evaluation()
    trainer_inner_cv.model_init()
    trainer_inner_cv.model.to(torch.device("cpu"))
    return trainer_inner_cv


if __name__ == "__main__":
    img_path = args.img_path
    gt_path = args.gt_path
    max_number_process = gpu_count

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

    train_transforms = [
        compose_input_output_transform(input_transform=cvtransforms.Compose(train_input_transform_list)),
        ]

    base_dataset = mpImage_sorted_by_patient_dataset(img_dir=args.datapath,
                                                     multi_label_gt_path=gt_path,
                                                     transform=None)

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
    # cv_split_list = nfold_cross_validation(len(train_dataset), n_fold=2)
    # cv_split_list = nfold_cross_validation(4, n_fold=2)
    cv_split_list = leave_one_out_cross_validation(len(train_dataset))
    # cv_split_list = leave_one_out_cross_validation(2)

    nested_cv_list = []
    """
    nested_cv_list = [([(train_idx00, val_idx00), (train_idx01, val_idx01),...], test_idx0),
                      ([(train_idx10, val_idx10), (train_idx01, val_idx01),...], test_idx1),
                      ......]
    inner_cv: choosing the best model hyperparameters, 
    """

    for cv_split in cv_split_list:
        outer_train, outer_test = cv_split
        inner_cv_split_list = leave_one_out_cross_validation(len(outer_train))
        inner_cv_split_outer_idx_list = []
        for inner_cv_split in inner_cv_split_list:
            inner_cv_train_idx, inner_val_idx = inner_cv_split
            inner_cv_split_outer_idx_list.append((outer_train[inner_cv_train_idx], outer_train[inner_val_idx]))
        nested_cv_list.append((inner_cv_split_outer_idx_list, outer_test))

    # running_states = ["train", "val", "test"]
    inner_running_states = ["train", "val"]
    outer_n_fold = len(cv_split_list)

    # cv_data_loaders = [{"train": DataLoader(dataset=train_dataset,
    #                                         batch_size=args.n_batch,
    #                                         sampler=SubsetRandomSampler(train_idx)),
    #                     "val": DataLoader(dataset=val_dataset,
    #                                       batch_size=args.n_batch,
    #                                       sampler=SubsetRandomSampler(val_idx))
    #                     } for train_idx, val_idx in cv_split_list]

    # Grid Search

    metric_list = ["f1_by_sample", "f1_by_label", "balanced_acc_by_label",
                   "auc_by_label", "ap_by_label", "fmax_by_label"
                   ]
    parameters_grid = {"epochs": [args.epochs],
                       "num_classes": [num_classes],
                       "multi_label": [True],
                       "n_fold": [outer_n_fold],
                       "performance_metrics_list": [metric_list],
                       # "device": [device],
                       "p_model": ["resnext101_32x8d"],
                       # "p_model": ["resnet18", "resnext101_32x8d"],
                       "p_weight": [True],
                       "feat_ext": [True],
                       "lr": [1e-3, 1e-5],
                       "wd": [1e-5, 1e-2],
                       "input_res": [(3, input_tensor_size[0], input_tensor_size[1])],
                       "out_list": [False, True]
                       }
    list_parameters = ParameterGrid(parameters_grid)


    for outer_nth_fold in range(outer_n_fold):
        print("{} outer {}th fold: {}".format("-" * 10, outer_nth_fold, "-" * 10))
        inner_cv, test_idx = nested_cv_list[outer_nth_fold]
        test_data_loader = DataLoader(dataset=val_dataset,
                                      batch_size=args.n_batch,
                                      sampler=SubsetRandomSampler(test_idx))
        outer_train_data_loader = DataLoader(dataset=train_dataset,
                                      batch_size=args.n_batch,
                                      sampler=SubsetRandomSampler(cv_split_list[outer_nth_fold][0]))
        parametric_model_list = []
        # parametric_scores
        # run with different hyper params set

        """
        Single thread run
        """
        single_thread_start_time = time.time()
        for idx, parameters in enumerate(list_parameters):

            specific_trainer = run_inner_cv(parameters=parameters,
                                            inner_cv=inner_cv,
                                            train_dataset=train_dataset,
                                            val_dataset=val_dataset,
                                            epochs=args.epochs,
                                            n_batch=args.n_batch,
                                            iterate_idx=idx,
                                            gpu_count=gpu_count
                                            )
            parametric_model_list.append(specific_trainer)
        single_time_elapsed = time.time()-single_thread_start_time
        print("Single thread running takes {:.0f}m {:.0f}s".format( single_time_elapsed // 60,
                                                               single_time_elapsed % 60))
        # """
        # Parallelism with joblib library
        # """
        # parallel_thread_start_time = time.time()
        # parametric_model_list = Parallel(n_jobs=gpu_count)(delayed(run_inner_cv)(parameters=parameters,
        #                                                                         inner_cv=inner_cv,
        #                                                                         train_dataset=train_dataset,
        #                                                                         val_dataset=val_dataset,
        #                                                                         epochs=args.epochs,
        #                                                                         n_batch=args.n_batch,
        #                                                                         iterate_idx=idx,
        #                                                                     gpu_count=gpu_count) for idx, parameters in enumerate(list_parameters))
        #
        # parallel_time_elapsed = time.time()-parallel_thread_start_time
        # print("Single thread running takes {:.0f}m {:.0f}s".format( single_time_elapsed // 60,
        #                                                        parallel_time_elapsed % 60))
        # TODO: which metric shd be used for picking best model?
        # parametric_scores[specific_trainer.model_name] = specific_trainer.performance_stat


            # parametric_model_list.append(specific_trainer)
    #     TODO: decide the best model with the list of parametric model

    #     Which metric shd be used for picking best model?



    out_df = base_dataset.multi_label_df
    label_list = base_dataset.label_name
    result_path = args.datapath + "patient_classify_result/"
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

    out_df.to_csv(result_path + 'result.csv', index=None, header=True)
