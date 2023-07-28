# Used to monitor memory (of gpu)
# https://github.com/Oldpan/Pytorch-Memory-Utils for reference

import torch
import torch.nn as nn
import numpy as np

def model_size(model: nn.Module, input_val: torch.Tensor, type_size: int=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : Number of params: {}'.format(model._get_name(), para))
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input_val.clone()
    # input_.requires_grad_(requires_grad=False)     # Deprecated in PyTorch 1.6.0?
    input_.requires_grad_(mode=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # print('Model {} : Number of intermediate variables without backward: {}'.format(model._get_name(), total_nums))
    # print('Model {} : Number of intermediate variables with backward: {}'.format(model._get_name(), total_nums*2))
    print('Model {} : intermediate variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermediate variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

