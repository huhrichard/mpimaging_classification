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
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='training for classification')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data', type=str, help='Path of data')
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (like regularization)')
parser.add_argument('--n_batch', default=1, type=int, help='weight decay (like regularization)')

using_gpu = torch.cuda.is_available()
print("Using GPU:", using_gpu)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

args = parser.parse_args()
print(args)


def model_training_and_evaluate_testing(epochs,
                                        cross_val_indices,
                                        test_indices,
                                        trainer):
    pass

def put_parameters_to_trainer(parameters):
    return trainer()

if __name__ == "__main__":
    img_path = args.datapath+"/img_path"
    gt_path = args.datapath+"/gt_path"
    parameters_grid = {"test_attr": 0}
    list_parameters = ParameterGrid(parameters_grid)

    # create dataset
    base_transform_list = [cvtransforms.Resize(size=(350, 350), interpolation='BILINEAR'),
                           cvtransforms.ToTensor(),
                           cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    transforms = [compose_input_output_transform(input_transform=cvtransforms.Compose(base_transform_list))]
    dataset = torch.utils.data.ConcatDataset([
                mpImage_dataset(img_path=args.datapath, gt_path=gt_path, transform=t) for t in transforms])

    # Split data into cross-validation_set and test_set
    cv_split_indices, test_indices = cross_validation_and_test_split(len(dataset))

    cv_data_samplers = [SubsetRandomSampler(cv_split_index) for cv_split_index in cv_split_indices]
    test_data_sampler = SubsetRandomSampler(test_indices)

    cv_data_loaders = [DataLoader(dataset=dataset, batch_size=args.n_batch, sampler=cv_data_sampler
                                  ) for cv_data_sampler in cv_data_samplers]

    test_data_loader = DataLoader(dataset=dataset, batch_size=args.n_batch, sampler=test_data_sampler)

    running_states = ["train", "val", "test"]
    # Grid Search
    parametric_model_list = []
    for parameters in list_parameters:
        trainer_list = []
        for nth_fold in range(len(cv_data_samplers)):
            trainer = put_parameters_to_trainer(parameters)
            for epoch in range(args.epochs):
                for running_state in running_states:
                    if running_state == "train":
                        train_splits = list(range(len(cv_data_samplers))).remove(nth_fold)
                        for train_split in random.shuffle(train_splits):
                            for batch_idx, (input, gt) in enumerate(cv_data_loaders[train_split]):
                                loss, predict = trainer.running_model(input, gt, running_state=running_state)
                    else:
                        if running_state == "val":
                            data_loader = cv_data_loaders[nth_fold]
                        elif running_state == "test":
                            data_loader = test_data_loader

                        for batch_idx, (input, gt) in enumerate(data_loader):
                            loss, predict = trainer.running_model(input, gt, running_state=running_state)

                    trainer.evaluation(running_state=running_state, epoch=epoch)









