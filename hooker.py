import numpy as np


def get_grad_from_optimizer(optimizer, world_size):
    grads = []
    count = 0
    for index, p_g in enumerate(optimizer.param_groups):
        print(count)
        count += 1
        grads.append([])
        for p in p_g['params']:
            grad = p.grad
            grads[index].append(grad / world_size)
    return grads
