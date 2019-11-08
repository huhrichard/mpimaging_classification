from utils.common_library import *
import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='training for classification')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data', type=str, help='Path of data')
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (like regularization)')
parser.add_argument('--n_batch', default=1, type=int, help='weight decay (like regularization)')



args = parser.parse_args()
print(args)



def model_training_and_evaluate_testing(epochs,
                                        cross_val_indices,
                                        test_indices,
                                        trainer):
    pass

def put_parameters_to_trainer(parameters):
    pass

if __name__ == "__main__":
    parameters_grid = {}
    list_parameters = ParameterGrid(parameters_grid)
    # Grid Search
    for parameters in list_parameters:
        trainer = put_parameters_to_trainer(parameters)



