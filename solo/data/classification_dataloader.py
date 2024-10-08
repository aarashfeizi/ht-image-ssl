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

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torchvision
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torchvision.datasets import STL10, ImageFolder, SVHN, OxfordIIITPet, DTD, INaturalist, CLEVRClassification
from solo.data.dataset_wrappers import FGVCAircraft, Cifar10, Cifar100
from solo.data.hf_datasets import Food101, Country211
from solo.data.inat18 import INAT18, INAT18_MEAN, INAT18_STD
from solo.data.imagefolder_missing_classes import ImageFolderMissingClasses
from solo.utils import misc
from solo.data.medmnist import PathMNIST, TissueMNIST
import numpy as np

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True

HOTELID_MEAN = (0.4620, 0.3980, 0.3292)
HOTELID_STD = (0.2619, 0.2529, 0.2460)


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    return pipeline


def prepare_transforms(dataset: str, min_scale_224=False) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    if min_scale_224:
        cifar_input_size = 224
        tinyimagenet_input_size = 224
        pathmnist_input_size = 224
    else:
        cifar_input_size = 32
        tinyimagenet_input_size = 64
        pathmnist_input_size = 28
    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=cifar_input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((cifar_input_size, cifar_input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    svhn_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=cifar_input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((cifar_input_size, cifar_input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    aircrafts_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    tinyimagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=tinyimagenet_input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(tinyimagenet_input_size),  # resize shorter
                # transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    pathmnist_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=pathmnist_input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(pathmnist_input_size),  # resize shorter
                # transforms.CenterCrop((28, 28)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    tissuemnist_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=pathmnist_input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[np.mean(IMAGENET_DEFAULT_MEAN)], std=[np.mean(IMAGENET_DEFAULT_STD)]),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(pathmnist_input_size),  # resize shorter
                # transforms.CenterCrop((28, 28)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=[np.mean(IMAGENET_DEFAULT_MEAN)], std=[np.mean(IMAGENET_DEFAULT_STD)]),
            ]
        ),
    }


    inat_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    inat18_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=INAT18_MEAN, std=INAT18_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=INAT18_MEAN, std=INAT18_STD),
            ]
        ),
    }


    pets_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(96, 96), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    dtd_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    country211_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    food101_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    hoteid_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=HOTELID_MEAN, std=HOTELID_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=HOTELID_MEAN, std=HOTELID_STD),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "food101": food101_pipeline,
        "country211": country211_pipeline,
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "svhn": svhn_pipeline,
        "aircrafts": aircrafts_pipeline,
        "pathmnist": pathmnist_pipeline,
        "tissuemnist": tissuemnist_pipeline,
        "inat": inat_pipeline,
        "inat18": inat18_pipeline,
        "pets": pets_pipeline,
        "dtd": dtd_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "tinyimagenet": tinyimagenet_pipeline,
        "hotelid-val": hoteid_pipeline,
        "hotelid-test": hoteid_pipeline,
        "hotels50k-test": hoteid_pipeline,
        "custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    download: bool = True,
    data_fraction: float = -1.0,
    test=False,
    is_classification=True,
    data_path=None
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder". 
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"


    assert dataset in ["food101", "country211", "cifar10", "cifar100", "stl10", "svhn", "tinyimagenet", "pathmnist", "tissuemnist", "inat", "inat18", "pets", "dtd", "eurosat", "aircrafts", "imagenet", "imagenet100", "hotelid-val", "hotelid-test", "hotels50k-test", "custom"]

    if dataset in ["cifar10", "cifar100"]:
        if dataset == 'cifar10':
            DatasetClass = Cifar10
        else:
            DatasetClass = Cifar100
            
        train_dataset = DatasetClass(
            train_data_path,
            split='train',
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            split='test',
            transform=T_val,
        )
    
    elif dataset == 'country211':
        train_dataset = Country211(
            split="train",
            transform=T_train
        )
        
        val_dataset = Country211(
            split="test",
            transform=T_val
        )
    
    elif dataset == 'food101':
        train_dataset = Food101(
            split="train",
            transform=T_train
        )
        
        val_dataset = Food101(
            split="validation",
            transform=T_val
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )
    
    elif dataset == 'svhn':
        train_dataset = SVHN(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = SVHN(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )
    
    elif dataset == 'aircrafts':
        if test:
            train_dataset = FGVCAircraft(
                train_data_path,
                split="trainval",
                transform=T_train,
            )
            val_dataset = FGVCAircraft(
                val_data_path,
                split="test",
                transform=T_val,
            )
        else:
            train_dataset = FGVCAircraft(
                train_data_path,
                split="train",
                transform=T_train,
            )
            val_dataset = FGVCAircraft(
                val_data_path,
                split="val",
                transform=T_val,
            )
    elif dataset == 'tinyimagenet':
        train_dataset = ImageFolder(
            train_data_path,
            transform=T_train,
        )
        val_dataset = ImageFolder(
            val_data_path,
            transform=T_val,
        )
    elif dataset == 'pathmnist':
        if test:
            train_dataset = PathMNIST(
                root=train_data_path,
                split="train",
                download=download,
                transform=T_train,
            )
            val_dataset = PathMNIST(
                root=val_data_path,
                split="test",
                download=download,
                transform=T_val,
            )
        else:
            train_dataset = PathMNIST(
                root=train_data_path,
                split="train",
                download=download,
                transform=T_train,
            )
            val_dataset = PathMNIST(
                root=val_data_path,
                split="val",
                download=download,
                transform=T_val,
            )
    elif dataset == 'tissuemnist':
        if test:
            train_dataset = TissueMNIST(
                root=train_data_path,
                split="train",
                download=download,
                transform=T_train,
            )
            val_dataset = TissueMNIST(
                root=val_data_path,
                split="test",
                download=download,
                transform=T_val,
            )
        else:
            train_dataset = TissueMNIST(
                root=train_data_path,
                split="train",
                download=download,
                transform=T_train,
            )
            val_dataset = TissueMNIST(
                root=val_data_path,
                split="val",
                download=download,
                transform=T_val,
            )

    elif dataset == 'inat':
        train_dataset = INaturalist(
            train_data_path,
            version="2021_train_mini",
            download=not os.path.exists(os.path.join(train_data_path, '2021_train_mini')),
            transform=T_train,
        )
        val_dataset = INaturalist(
            val_data_path,
            version="2021_valid",
            download=not os.path.exists(os.path.join(val_data_path, '2021_valid')),
            transform=T_val,
        )
    
    elif dataset == 'inat18':
        train_dataset = INAT18(
            root=data_path,
            ann_file=train_data_path,
            download=not os.path.exists(os.path.join(train_data_path, '2021_train_mini')),
            transform=T_train,
            is_train=True
        )
        val_dataset = INAT18(
            root=data_path,
            ann_file=val_data_path,
            transform=T_val,
            is_train=False
        )
    
    
    elif dataset == 'pets':
        train_dataset = OxfordIIITPet(
            train_data_path,
            split="trainval",
            download=True,
            transform=T_train,
        )
        val_dataset = OxfordIIITPet(
            val_data_path,
            split="test",
            download=True,
            transform=T_val,
        )
    
    elif dataset == 'dtd':
        train_dataset = DTD(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = DTD(
            val_data_path,
            split="test",
            download=True,
            transform=T_val,
        )
    

    elif dataset in ["imagenet", "imagenet100", "hotelid-val", "hotelid-test", "hotels50k-test", "custom"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)

        if dataset in ['hotelid-val', 'hotelid-test', "hotels50k-test"] and is_classification:
            val_dataset = ImageFolderMissingClasses(val_data_path, T_val,
                                                    classes=train_dataset.classes,
                                                    class_to_idx=train_dataset.class_to_idx)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
    data_fraction: float = -1.0,
    auto_augment: bool = False,
    subsample_by:int = 1,
    test=False,
    min_scale_224=False,
    data_path=None
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    T_train, T_val = prepare_transforms(dataset, min_scale_224)
    if auto_augment:
        T_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.25,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
        test=test,
        data_path=data_path
    )

    if subsample_by > 1:
        train_dataset = misc.subsample_dataset(train_dataset, subsample_by=subsample_by)
        val_dataset = misc.subsample_dataset(val_dataset, subsample_by=subsample_by)

    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader
