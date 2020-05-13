import torch
from torchvision import datasets, transforms

from torchutils.experiment import DataLoaders


def cifar10_loader(path: str, batch_size: int = 128, **kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    training_data = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
        )
    train_loader = torch.utils.data.DataLoader(
        training_data,
        shuffle=True,
        batch_size=batch_size
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_data = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
        )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=batch_size
        )
    return DataLoaders(train=train_loader, test=test_loader)


def cifar100_loader(path: str, batch_size: int = 128, **kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    training_data = datasets.CIFAR100(
        root=path, train=True, download=True, transform=transform_train
        )
    train_loader = torch.utils.data.DataLoader(
        training_data,
        shuffle=True,
        batch_size=batch_size
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_data = datasets.CIFAR100(
        root=path, train=False, download=True, transform=transform_test
        )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=batch_size
        )
    return DataLoaders(train=train_loader, test=test_loader)


def mnist_loader(path: str, batch_size: int = 128, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs)
    return DataLoaders(train=train_loader, test=test_loader)