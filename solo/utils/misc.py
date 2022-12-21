# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from solo.data.h5_dataset import H5Dataset
from timm.models.helpers import group_parameters
from timm.optim.optim_factory import _layer_map
from tqdm import tqdm
import faiss

def _1d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite()


def _2d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite().all(dim=1)


def _single_input_filter(tensor: torch.Tensor) -> Tuple[torch.Tensor]:
    if len(tensor.size()) == 1:
        filter_func = _1d_filter
    elif len(tensor.size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensor)
    tensor = tensor[selected]

    return tensor, selected


def _multi_input_filter(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor]:
    if len(tensors[0].size()) == 1:
        filter_func = _1d_filter
    elif len(tensors[0].size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensors[0])
    for tensor in tensors[1:]:
        selected = torch.logical_and(selected, filter_func(tensor))
    tensors = [tensor[selected] for tensor in tensors]

    return tensors, selected


def filter_inf_n_nan(tensors: List[torch.Tensor], return_indexes: bool = False):
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (List): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """

    if isinstance(tensors, torch.Tensor):
        tensors, selected = _single_input_filter(tensors)
    else:
        tensors, selected = _multi_input_filter(tensors)

    if return_indexes:
        return tensors, selected
    return tensors


class FilterInfNNan(nn.Module):
    def __init__(self, module):
        """Layer that filters out inf and nans from any tensor.
        This is usefull when there are instability issues,
        which cause a small number of values to go bad.

        Args:
            tensor (List): tensor to remove nans and infs from.

        Returns:
            torch.Tensor: filtered view of the tensor without nans or infs.
        """
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        out = filter_inf_n_nan(out)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "module":
                raise AttributeError()
            return getattr(self.module, name)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def norm_cdf(x):
        """Computes standard normal cumulative distribution function"""

        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logging.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


@torch.no_grad()
def concat_all_gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    if dist.is_available() and dist.is_initialized():
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    return tensor


def compute_dataset_size(
    dataset: Optional[str] = None,
    train: Optional[bool] = True,
    data_path: Optional[str] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[bool] = False,
    data_fraction: Optional[float] = -1,
):
    """Utility function to get the dataset size. If using cifar or stl,
    provide dataset and the train flag.
    E.g., compute_dataset_size(dataset='cifar10', train=True/False).
    When using an ImageFolder dataset, just provide the path to the folder and
    specify if it has labels or not with the no_labels flag.

    Args:
        dataset (Optional[str]): dataset size for predefined datasets
            [cifar10, cifar100, stl10]. Defaults to None.
        train (Optional[bool]): train dataset flag. Defaults to True.
        data_path (Optional[str]): path to the folder. Defaults to None.
        data_format (Optional[str]): format of the data, either "image_folder" or "h5".
            Defaults to "image_folder".
        no_labels (Optional[bool]): if the dataset has no labels. Defaults to False.
        data_fraction (Optional[float]): amount of data to use. Defaults to -1.

    Returns:
        int: size of the dataset
    """

    DATASET_SIZES = {
        "cifar10": {"train": 50_000, "val": 10_000},
        "cifar100": {"train": 50_000, "val": 10_000},
        "stl10": {"train": 105_000, "val": 8_000},
    }
    size = None

    if dataset is not None:
        size = DATASET_SIZES.get(dataset.lower(), {}).get("train" if train else "val", None)

    if data_format == "h5":
        size = len(H5Dataset(dataset, data_path))

    if size is None:
        if no_labels:
            size = len(os.listdir(data_path))
        else:
            size = sum(
                len(os.listdir(os.path.join(data_path, class_))) for class_ in os.listdir(data_path)
            )

    if data_fraction != -1:
        size = int(size * data_fraction)

    return size


def make_contiguous(module):
    """Make the model contigous in order to comply with some distributed strategies.
    https://github.com/lucidrains/DALLE-pytorch/issues/330
    """

    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Adapted from https://github.com/facebookresearch/mae.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = generate_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def generate_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # Adapted from https://github.com/facebookresearch/mae.

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def generate_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Adapted from https://github.com/facebookresearch/mae.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def param_groups_layer_decay(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Tuple[str] = (),
    layer_decay: float = 0.75,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """

    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, "group_matcher"):
        # FIXME interface needs more work
        layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    else:
        # fallback
        layer_map = _layer_map(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


def remove_bias_and_norm_from_weight_decay(parameter_groups: List[Dict]):
    out = []
    for group in parameter_groups:
        # parameters with weight decay
        decay_group = {k: v for k, v in group.items() if k != "params"}

        # parameters without weight decay
        no_decay_group = {k: v for k, v in group.items() if k != "params"}
        no_decay_group["weight_decay"] = 0
        group_name = group.get("name", None)
        if group_name:
            no_decay_group["name"] = group_name + "_no_decay"

        # split parameters into the two lists
        decay_params = []
        no_decay_params = []
        for param in group["params"]:
            if param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # add groups back
        if decay_params:
            decay_group["params"] = decay_params
            out.append(decay_group)
        if no_decay_params:
            no_decay_group["params"] = no_decay_params
            out.append(no_decay_group)
    return out


def omegaconf_select(cfg, key, default=None):
    """Wrapper for OmegaConf.select to allow None to be returned instead of 'None'."""
    value = OmegaConf.select(cfg, key, default=default)
    if value == "None":
        return None
    return value

def train_emb_model(cfg, model, train_loader, val_loader=None):
    import wandb
    OPTS = {'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW}
    
    LOSSES = {'mse': torch.nn.MSELoss}

    epochs = cfg.emb_model.epochs
    loss_fn = LOSSES[cfg.emb_model.loss]()
    optimizer = OPTS[cfg.emb_model.opt](model.parameters(),
                                lr=cfg.emb_model.lr,
                                weight_decay=cfg.emb_model.weight_decay)

    def train_one_step(current_epoch):
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'{current_epoch}/{epochs}') as t:
            for idx, batch in enumerate(train_loader, start=1):
                _, X, _ = batch
                X = X.cuda()
                X_pred = model(X)

                loss = loss_fn(X_pred, X)

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.update()

        return total_loss / len(train_loader)
    
    def val(current_epoch):
        total_loss = 0
        with tqdm(total=len(val_loader), desc=f'{current_epoch}/{epochs}') as t:
            for idx, batch in enumerate(val_loader, start=1):
                _, X, targets = batch
                X = X.cuda()
                X_pred = model(X)

                loss = loss_fn(X_pred, X)

                total_loss += loss.item()

                t.update()
                
        return total_loss / len(val_loader)
    
    for epoch in range(1, epochs + 1):
        wandb_dict = {}
        train_loss = train_one_step(epoch)
        wandb_dict[f'emb_model/train_{cfg.emb_model.loss}_loss'] = train_loss

        if val_loader is not None:
            val_loss = val(epoch)
            wandb_dict[f'emb_model/val_{cfg.emb_model.loss}_loss'] = val_loss
        
        wandb.log(wandb_dict)
    
    return model
        

def get_embeddings(model, dataloader):
    embs = []

    with tqdm(total=len(dataloader), desc='Getting embeddings...') as t:
        for idx, batch in enumerate(dataloader):
            _, X, targets = batch
            X = X.cuda()
            batch_emb = model(X)
            embs.append(batch_emb.detach().cpu().numpy())
            t.update()

    embs = np.concatenate(embs)
    return embs

def get_sim_matrix(embeddings, k=1000):
    d = embeddings.shape[-1]
    cpu_index = faiss.IndexFlatL2(d)

    try:
        final_index = faiss.index_cpu_to_all_gpus(cpu_index)

        final_index.add(embeddings)
        print('Using GPU for NN!! Thanks FAISS! :)')
        print(final_index.ntotal)
    except:
        print('No gpus for faiss! :( ')
        final_index = cpu_index
        final_index.add(embeddings)

    D, I = final_index.search(embeddings, k) # actual search
    
    return D, I

def load_npy(path):
    return np.load(path)

def save_npy(data, path):
    np.save(path, data)
    return True

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)