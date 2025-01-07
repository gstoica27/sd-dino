import pdb
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_identity_perturbations(dim, scale):
    transform = torch.eye(dim)
    perturbs = torch.randn(dim) * scale
    idxs = range(len(transform))
    transform[idxs, idxs] = perturbs
    return transform


def random_invertible_transform(dim, scale):
    transform = torch.randn(dim, dim) * scale
    return transform


def random_orthogonal_transform(dim, scale):
    transform = torch.randn(dim, dim)
    q, r = torch.qr(transform)
    return q * scale


def random_permutation_transform(dim, scale):
    transform = torch.eye(dim) * scale
    idxs = torch.randperm(dim)
    transform = transform[idxs]
    return transform


def custom_transform(path):
    transform = torch.load(path)
    if isinstance(transform, dict):
        transform = transform['weight']
    if len(transform.size()) == 3: # heads x dim x dim
        # pdb.set_trace()
        dim = transform.shape[0] * transform.shape[1]
        out = torch.zeros((dim, dim), dtype=transform.dtype)
        for block_idx in range(transform.shape[0]):
            start = block_idx * transform.shape[1]
            end = start + transform.shape[1]
            out[start:end, start:end] = transform[block_idx]
        # pdb.set_trace()
    else:
        out = transform
    return out


def custom_inverse_transform(path):
    transform = custom_transform(path)
    return torch.linalg.pinv(transform)


def compute_random_transform(dim, scale, transform_type='identity', load_path=None):
    if transform_type == 'random_identity':
        return random_identity_perturbations(dim, scale)
    elif transform_type == 'random_invertible':
        return random_invertible_transform(dim, scale)
    elif transform_type == 'random_orthogonal':
        return random_orthogonal_transform(dim, scale)
    elif transform_type == 'random_permutation':
        return random_permutation_transform(dim, scale)
    else:
        raise ValueError('Unknown transform type: {}'.format(transform_type))


def get_random_transform(transform_config, feature_info):
    # pdb.set_trace()
    num_heads = feature_info['num_heads']
    total_dim = feature_info['dim']
    dim_per_head = total_dim // num_heads
    full_transform = torch.zeros(total_dim, total_dim)
    for i in range(num_heads):
        transform = compute_random_transform(
            dim_per_head, 
            transform_config.get('scale', 1.), 
            transform_config['type']
        )
        block_start = i * dim_per_head
        block_end = block_start + dim_per_head    
        full_transform[block_start:block_end, block_start:block_end] = transform
    return full_transform


def get_custom_transform(transform_type, load_path=None):
    if transform_type == 'custom':
        assert load_path is not None
        return custom_transform(load_path)
    elif transform_type == 'custom_inverse':
        assert load_path is not None
        return custom_inverse_transform(load_path)
    else:
        raise ValueError('Unknown transform type: {}'.format(transform_type))
        
    
class Transform(nn.Module):
    def __init__(self, transform_config, feature_info, seed=42):
        super(Transform, self).__init__()
        # set seeds each time
        self.reset_seed(seed)
        print("Transform config: ", transform_config)
        print("Feature config: ", feature_info)
        self.transform = self.create_transform(transform_config, feature_info)
        # if transform_config['add_bias']:
        #     self.bias = torch.randn(feature_info['dim']) * transform_config['bias_scale']
        # else:
        #     self.bias = torch.zeros([feature_info['dim']])
    
    def create_transform(self, transform_config, feature_info):
        if transform_config['type'] in [
            'random_identity', 'random_invertible', 'random_orthogonal', 
            'random_permutation'
        ]:
            transform = get_random_transform(transform_config, feature_info)
        elif transform_config['type'] in ['custom', 'custom_inverse']:
            transform = get_custom_transform(
                transform_config['type'], transform_config['load_path']
            )
        else:
            raise ValueError('Unknown transform type: {}'.format(transform_config['type']))
        return transform
    
    def reset_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def forward(self, x):
        if self.transform.device != x.device:
            self.transform = self.transform.to(x.device)
        return x @ self.transform #+ self.bias