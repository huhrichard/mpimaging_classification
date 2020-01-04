import torch
import subprocess

using_gpu = torch.cuda.is_available()
print("Using GPU: ", using_gpu)

gpu_count = torch.cuda.device_count()
print("Avaliable GPU:", gpu_count)

# Device configuration
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
# Convert lines into a dictionary
# gpu_memory = [int(x) for x in result.strip().split('\n')]
# gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
print(result)