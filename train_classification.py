from utils.common_library import *
import argparse
from sklearn.model_selection import ParameterGrid
from utils.trainer import trainer
from utils.preprocess_data_loader import *
from utils.preprocess_data_transform import compose_input_output_transform
from cvtorchvision import cvtransforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import random
from utils.model import simple_transfer_classifier
from torch.autograd import Variable
import time
from utils.postprocessing_visualization import compare_model
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='training for classification')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data/', type=str, help='Path of data')
parser.add_argument('--img_path', default='data/MPM/', type=str, help='Path of data')
parser.add_argument('--gt_path', default='data/TMA2_MPM_Summary.csv', type=str, help='File of the groundtruth')
parser.add_argument('--lr', '--learning_rate', default=1e-7, type=float, help='learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (like regularization)')
parser.add_argument('--n_batch', default=1, type=int, help='weight decay (like regularization)')

using_gpu = torch.cuda.is_available()
print("Using GPU: ", using_gpu)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

args = parser.parse_args()
print(args)

input_tensor_size = (500, 500)

def model_training_and_evaluate_testing(epochs,
                                        cross_val_indices,
                                        test_indices,
                                        trainer):
    pass

def put_parameters_to_trainer(parameters, num_classes, device):

    model = simple_transfer_classifier(num_classes=num_classes,
                                       input_size=(3,input_tensor_size[0],input_tensor_size[1]),
                                       feature_extracting=False,
                                       ).to(device)
    new_trainer = trainer(model=model,
                            model_name="pretrained_1Linear",
                            optimizer=torch.optim.Adam(lr=args.lr, weight_decay=args.wd, params=model.parameters()),
                            n_batches=args.n_batch,
                            total_epochs=args.epochs,
                            lr_scheduler_list=[],
                            loss_function=nn.BCELoss())
    return new_trainer

def parameters_dict_to_model_name(parameters_dict):
    pass

if __name__ == "__main__":
    img_path = args.img_path
    gt_path = args.gt_path
    parameters_grid = {"test_attr": [0],
                       # "feature_extracting": [True, False]
                       }
    list_parameters = ParameterGrid(parameters_grid)

    # create dataset
    test_input_transform_list = [cvtransforms.Resize(size=input_tensor_size, interpolation='BILINEAR'),
                                 cvtransforms.ToTensor(),
                                 cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    train_val_input_transform_list = [cvtransforms.Resize(size=input_tensor_size, interpolation='BILINEAR'),
                                 # cvtransforms.RandomHorizontalFlip(),
                                 # cvtransforms.RandomVerticalFlip(),
                                 # cvtransforms.RandomRotation(90),
                                 cvtransforms.ToTensor(),
                                 cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    # hflip_input_transform_list = [cvtransforms.Resize(size=input_tensor_size, interpolation='BILINEAR'),
    #                              cvtransforms.RandomHorizontalFlip(p=1),
    #                              cvtransforms.ToTensor(),
    #                              cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]



    # output_transform =

    # CV_data set
    train_val_transforms = [compose_input_output_transform(input_transform=cvtransforms.Compose(train_val_input_transform_list)),
                  ]

    train_val_dataset = torch.utils.data.ConcatDataset([
                mpImage_sorted_by_image_dataset(img_dir=args.datapath, gt_path=gt_path, transform=t) for t in train_val_transforms])


    num_classes = train_val_dataset[0]["gt"].shape[0]
    # Split data into cross-validation_set and test_set
    cv_split_indices, test_indices = cross_validation_and_test_split(len(train_val_dataset))
    print(cv_split_indices, test_indices)

    cv_data_samplers = [SubsetRandomSampler(cv_split_index) for cv_split_index in cv_split_indices]

    cv_data_loaders = [DataLoader(dataset=train_val_dataset, batch_size=args.n_batch, sampler=cv_data_sampler
                                  ) for cv_data_sampler in cv_data_samplers]

    # test data loader
    test_transforms = [compose_input_output_transform(input_transform=cvtransforms.Compose(train_val_input_transform_list)),
                       ]
    test_dataset = torch.utils.data.ConcatDataset([
                mpImage_sorted_by_image_dataset(img_dir=args.datapath, gt_path=gt_path, transform=t) for t in test_transforms])

    test_data_sampler = SubsetRandomSampler(test_indices)

    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.n_batch, sampler=test_data_sampler)

    running_states = ["train", "val", "test"]
    # Grid Search
    parametric_model_list = []
    for parameters in list_parameters:
        trainer_list = []
        for nth_fold in range(len(cv_data_samplers)):
            print("{} {}th fold: {}".format("-" * 10, nth_fold, "-" * 10))
            specific_trainer = put_parameters_to_trainer(parameters, num_classes, device)
            for epoch in range(args.epochs):
                print("="*30)
                print("{} {}th epoch running: {}".format("="*10, epoch, "="*10))
                epoch_start_time = time.time()
                for running_state in running_states:
                    if running_state == "train":
                        train_splits = list(range(len(cv_data_samplers)))
                        train_splits.remove(nth_fold)
                        random.shuffle(train_splits)
                        for train_split in train_splits:
                            for batch_idx, data in enumerate(cv_data_loaders[train_split]):
                                input = data['input']
                                # print("input mean and sd: ", input.mean(), input.std())
                                gt = data['gt']
                                input = Variable(input).float().to(device)
                                gt = Variable(gt).float().to(device)
                                loss, predict = specific_trainer.running_model(input, gt, epoch=epoch, running_state=running_state)
                    else:
                        if running_state == "val":
                            data_loader = cv_data_loaders[nth_fold]
                        elif running_state == "test":
                            data_loader = test_data_loader

                        for batch_idx, data in enumerate(data_loader):
                            input = data['input']
                            gt = data['gt']
                            input = Variable(input).float().to(device)
                            gt = Variable(gt).float().to(device)
                            loss, predict = specific_trainer.running_model(input, gt, epoch=epoch, running_state=running_state)

                    specific_trainer.evaluation(running_state=running_state, epoch=epoch)

                time_elapsed = time.time()-epoch_start_time
                print("{}{}th epoch running time cost: {:.0f}m {:.0f}s".format("-"*5, epoch, time_elapsed // 60, time_elapsed % 60))

            trainer_list.append(specific_trainer)
        parametric_model_list.append(trainer_list)

    compare_model(parametric_model_list, args.datapath+"result/")







