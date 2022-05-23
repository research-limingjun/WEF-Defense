"""
======================
@author:Mr.li
@time:2022/2/10:19:33
@email:1035626671@qq.com
======================
"""
import torch
from copy import deepcopy
import torch.optim as optim
import numpy as np


def get_n_params(model):
    """return the number of parameters in the model"""

    n_params = sum([np.prod(tensor.size()) for tensor in list(model.parameters())])
    return n_params


def get_std(model_A, model_B):
    """get the standard deviation at iteration 2 with the proposed heuristic"""

    list_tens_A = [tens_param.detach() for tens_param in list(model_A.parameters())]
    list_tens_B = [tens_param.detach() for tens_param in list(model_B.parameters())]



    sum_abs_diff = 0

    for tens_A, tens_B in zip(list_tens_A, list_tens_B):
        sum_abs_diff += torch.sum(torch.abs(tens_A - tens_B))

    std = sum_abs_diff / get_n_params(model_A)

    return [0, std.item()]