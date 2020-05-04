import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from torchutils.dataloaders import cifar10_loader
from torchutils.experiment import Experiment, Config
from torchutils.tuning import scan_lr


def config():
    return Config(
        max_epochs=100,
        device=torch.device("cpu")
    )


def experiment(lr=1e-4):
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    optimizer = optim.SGD(resnet18.parameters(), lr=lr)
    return Experiment(
        model=resnet18,
        optimizer=optimizer,
        config=config(),
        loss_fn=nn.CrossEntropyLoss(),
        lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    )

def _main():
    data = cifar10_loader(path='../data')
    exp = experiment()
    res = scan_lr(exp, data, min_lr=0.001, max_lr=1, n_epochs=2)
    lr_plot(*res)


if __name__ == '__main__':
    _main()