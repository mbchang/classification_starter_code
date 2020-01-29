"""
You can either create a new dataloader, or filter when you sample

You don't even need to augment the data in a weird way, you can just sample at a different frequency
"""

from __future__ import print_function
import argparse
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from starter_code.log import log_string, ClassificationLogger
from starter_code.utils import split_groups

root = './classification/runs'


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
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    # get the classes in test_loader
    classes_correct = {label: 0 for label in set(test_loader.dataset.test_labels.numpy())}
    classes_total = {label: 0 for label in set(test_loader.dataset.test_labels.numpy())}

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            total_correct, correct_per_class, total_per_class = compute_correct(pred, target)
            correct += total_correct

            for label in correct_per_class:
                classes_correct[label] += correct_per_class[label]
                classes_total[label] += total_per_class[label]

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    class_accuracy = OrderedDict({
        label: float(classes_correct[label])/classes_total[label] for label in sorted(classes_correct)})

    return class_accuracy

def evaluate(args, model, device, test_loader, espoch, logger):
    class_accuracy = test(args, model, device, test_loader, epoch)
    print(log_string(OrderedDict(class_accuracy)))
    plot_class_accuracy(class_accuracy, 
        fname=os.path.join(logger.logdir, 'mnist_class_accuracies_epoch{}.png'.format(epoch)))

def plot_class_accuracy(class_accuracy, fname):
    labels, accuracies = zip(*class_accuracy.items())  # note that these are normalized
    fig, ax = plt.subplots()
    ax.bar(labels, accuracies, align='center', alpha=0.5)
    ax.set_xticks(range(len(labels)), map(int, labels))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Class')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def compute_correct(pred, target):
    pred = pred[:20]
    target = target[:20]

    num_equal = pred.eq(target.view_as(pred)).squeeze(-1)
    total_correct = num_equal.sum().item()

    ##################################################

    sorted_target, sorted_indices = torch.sort(target)
    sorted_num_equal = num_equal[sorted_indices]

    sorted_target = sorted_target.numpy()
    sorted_num_equal = sorted_num_equal.numpy()

    group_splitter = split_groups(sorted_target)
    target_groups = group_splitter(sorted_target)
    num_equal_groups = group_splitter(sorted_num_equal)

    correct_per_class = dict()
    total_per_class = dict()
    for target_group, num_equal_group in zip(target_groups, num_equal_groups):
        label_id = target_group[0]
        correct_per_class[label_id] = sum(num_equal_group)
        total_per_class[label_id] = len(num_equal_group)

    return total_correct, correct_per_class, total_per_class


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')


    parser.add_argument('--printf', action='store_true')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    args.expname = 'mnist'
    args.subroot = 'debug'
    logger = ClassificationLogger(args)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(args.epochs):
        evaluate(args, model, device, test_loader, epoch, logger)
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
    evaluate(args, model, device, test_loader, logger, epoch, logger)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
