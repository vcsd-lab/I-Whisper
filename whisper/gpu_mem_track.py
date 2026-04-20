import gc
import os
import time
from math import log2
from typing import Union
import logging

import torch
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
import numpy as np

dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.uint32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.uint16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
        logging_fn(Union[Logger, str], default 'EXELOG/'): Logging method or path to save the gpu memory track log
    """
    def __init__(self, detail=True, verbose=False, device=0, logging_fn:Union[logging.Logger, str]="EXELOG/"):
        self.print_detail = detail
        self.last_tensor_sizes = set()
        if isinstance(logging_fn, str):
            self.gpu_profile_fn = logging.getLogger("main" if __name__ == "__main__" else __name__)
            fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
            # create file handlers
            file_handler = logging.FileHandler(os.path.join(logging_fn, f'GPUMEM_TRACK_{int(time.time())}.log'), mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            self.gpu_profile_fn.addHandler(file_handler)
            self.gpu_profile_fn.setLevel(logging.DEBUG)
        else:
            self.gpu_profile_fn = logging_fn
        self.verbose = verbose
        self.begin = True
        self.device = device

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        return torch.cuda.memory_allocated() / 1024**2

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def track(self)->list[str]:
        """
        Track the GPU memory usage
        """
        return_str = []

        usage = f"{self.get_allocate_usage():<7.1f}MB"
        str_to_log = f"->| GPU Memory Track                  | CUDA MEMORY Allocated:{usage:<38}|<-"
        if self.begin:
            return_str.append(str_to_log)
            self.begin = False

        return_str.append("-"*115)
        return_str.append(str_to_log)
        return_str.append("-"*115)
        if self.print_detail is True:
            ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
            new_tensor_sizes = {(type(x),
                                tuple(x.size()),
                                ts_list.count((x.size(), x.dtype)),
                                np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                x.dtype) for x in self.get_tensors()}
            if new_tensor_sizes != self.last_tensor_sizes:
                for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                    return_str.append(f'+ | {str(n):>5} * Size:{str(s):<20} | Memory: {str(m*n)[:6]:>6} M | {str(t):>40} | {data_type}')
                for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                    return_str.append(f'- | {str(n):>5} * Size:{str(s):<20} | Memory: {str(m*n)[:6]:>6} M | {str(t):>40} | {data_type}')
                return_str.append("-"*115)

            self.last_tensor_sizes = new_tensor_sizes
        return return_str