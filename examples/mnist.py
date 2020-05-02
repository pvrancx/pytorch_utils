import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchutils.callbacks import CallbackHandler, ModelSaverCallback, LoggerCallback
from torchutils.experiment import DataLoaders, Config, Experiment
from torchutils.train import fit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_data(batch_size=128, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs)
    return DataLoaders(train=train_loader, test=test_loader)


def config():
    return Config(
        max_epochs=100,
        device=torch.device("cpu")
    )


def experiment(lr=1e-4):
    network = Net()
    optimizer = optim.Adadelta(network.parameters(), lr=lr)
    return Experiment(
        model=network,
        optimizer=optimizer,
        config=config(),
        loss_fn=F.nll_loss,
        lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    )


def callbacks():
    return CallbackHandler([
        LoggerCallback(),
        ModelSaverCallback('.', frequency=10)]
    )


def _main():
    data = get_data()
    exp = experiment()
    fit(exp, data, callbacks())


if __name__ == '__main__':
    _main()
