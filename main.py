'''Train CIFAR10 with PyTorch.'''
import socket

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from Trainer import Trainer

from models import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    # Distributed
    parser.add_argument('--rank', type=int, help='the rank of part')
    parser.add_argument('--size', type=int, help='size of client in the world')
    parser.add_argument('--port', type=int, help='self port')
    parser.add_argument('--n_ip', type=str, help='next ring allreduce server ip')
    parser.add_argument('--n_port', type=int, help='next ring allreduce server port')

    args = parser.parse_args()

    # Distributed
    rank = args.rank
    world_size = args.size
    self_port = args.port
    next_addr = (args.n_ip, args.n_port)
    # Distributed End

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    # Distributed
    sampler = torch.utils.data.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    # Distributed End

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=(sampler is None), sampler=sampler, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Init distributed
    _socket_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _socket_recv.bind(('0.0.0.0', self_port))
    _socket_recv.listen(5)

    socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if rank == 0:
        socket_send.connect(next_addr)
        print('Rank %d client connected' % rank)

        socket_recv, recv_addr = _socket_recv.accept()
        print('Rank %d server connected' % rank, recv_addr)
    else:
        socket_recv, recv_addr = _socket_recv.accept()
        print('Rank %d server connected' % rank, recv_addr)

        socket_send.connect(next_addr)
        print('Rank %d client connected' % rank)

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    net = net.to(DEVICE)
    if DEVICE == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    # socket_send.setsockopt(
    #     socket.SOL_SOCKET,
    #     socket.SO_SNDBUF,
    #     1024 * 15000
    # )

    trainer = Trainer(net, rank, world_size, trainloader, testloader, optimizer, criterion, socket_send, socket_recv)
    trainer.sync_model()
    for epoch in range(start_epoch, start_epoch + 200):
        trainer.train(epoch)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
