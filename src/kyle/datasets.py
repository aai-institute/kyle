import torch
from kornia.enhance import denormalize
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

# see https://github.com/akamaster/pytorch_resnet_cifar10
resnet_normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def resnet_denormalize_transform(data: Tensor):
    is_batch = len(data.shape) == 4
    if not is_batch:
        data = data[None, :]  # transform only works on batches
    result = denormalize(
        data,
        tensor(resnet_normalize_transform.mean),
        tensor(resnet_normalize_transform.std),
    )
    if not is_batch:
        result = result[0]
    return result


def get_cifar10_dataloader(path: str, train=False):
    dataset = CIFAR10(
        path,
        train=train,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), resnet_normalize_transform]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    return dataloader


def get_cifar10_dataset(path: str, train=False):
    dataset = CIFAR10(
        path,
        train=train,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), resnet_normalize_transform]
        ),
    )
    images = []
    targets = []
    # Quick hack, can't find a nice way of doing that. Datasets cannot be sliced and we need the transform
    # Alternative is to retrieve the .data array and do the transformations and reshaping ourselves but this is brittle
    for image, target in dataset:
        images.append(image)
        targets.append(target)
    return torch.stack(images), torch.tensor(targets)
