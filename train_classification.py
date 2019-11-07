from utils.common_library import *
import argparse

parser = argparse.ArgumentParser(description='training for classification')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--datapath', default='data', type=str, help='Path of data')
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (like regularization)')


args = parser.parse_args()
print(args)

