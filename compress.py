import numpy as np
import torch


def compress(value):
    np_value = value.numpy()
    shape = np_value.shape
    flattened_value = np_value.flatten()
    zero_mask = np.zeros(flattened_value.shape, dtype=np.int)
    zero_mask[flattened_value == 0] = 1
    vals_position = np.where(zero_mask == 0)
    zero_mask_bitmap = np.packbits(zero_mask)
    compressed_data = flattened_value[vals_position]
    return shape, compressed_data, zero_mask_bitmap


def uncompress(shape, compressed_data, zero_mask_bitmap):
    total_length = np.prod(shape)
    uncompressed_data = np.zeros([total_length])
    zero_mask = np.unpackbits(zero_mask_bitmap)[:total_length]
    uncompressed_data[zero_mask == 0] = compressed_data
    uncompressed_data = uncompressed_data.reshape(shape)
    return torch.from_numpy(uncompressed_data)
