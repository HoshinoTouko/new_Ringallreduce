import time

from utils import progress_bar

import os
import torch

import hooker
import pickle
import copy

from queue import Queue


class Trainer:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    ENDFLAG = b'BINARY_END'
    RECV_BUFFER = 150000 * 1024

    def __init__(self, net, rank, world_size, trainloader, testloader, optimizer, criterion, socket_send, socket_recv):
        self.net = net
        self.rank = rank
        self.world_size = world_size
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.socket_send = socket_send
        self.socket_recv = socket_recv

        self.sticky_cache = Queue()
        self.best_acc = 0

    # Test tool
    def print_params(self, model_to_print):
        for name, param in model_to_print.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                break
        input()

    def send(self, data):
        data_to_send = pickle.dumps(data)
        self.socket_send.send(data_to_send + self.ENDFLAG)

    def recv(self):
        if not self.sticky_cache.empty():
            return self.sticky_cache.get()

        data = b''
        while not data.endswith(self.ENDFLAG):
            data += self.socket_recv.recv(self.RECV_BUFFER)

        data_to_resolve = data.split(self.ENDFLAG)[:-1]

        data_to_return = None
        for index, obj in enumerate(data_to_resolve):
            obj = pickle.loads(obj)
            if index == 0:
                data_to_return = obj
            else:
                self.sticky_cache.put(obj)
        return data_to_return

    def sync_model(self):
        print('Rank %d start to sync model' % self.rank)
        if self.rank == 0:
            self.send(self.net.state_dict())
            print('Rank %d, send model end' % self.rank)
            return

        params = self.recv()
        self.net.load_state_dict(params)
        print('Recv model end')
        if self.rank != 0 and self.rank != self.world_size - 1:
            self.send(self.net.state_dict())
            print('Rank %d, send model end' % self.rank)

    def cut(self, grads):
        workload = int(len(grads) / self.world_size) + 1
        return [
            grads[workload * offset:workload * (offset + 1)]
            for offset in range(self.world_size)
        ]

    def ring_allreduce(self, grads):
        works = self.cut(grads[0])

        workload_id = self.rank
        for _ in range(self.world_size - 1):
            # Send data
            self.send(works[workload_id])
            # print('1. Send workload id %d' % workload_id)
            workload_id -= 1

            # Recv data
            data_to_merge = self.recv()
            # print('1. Recv workload id %d' % workload_id)
            assert len(data_to_merge) == len(works[workload_id])

            for index, item in enumerate(data_to_merge):
                works[workload_id][index] += item

        if self.rank == 0:
            workload_id += self.world_size

        for _ in range(self.world_size - 1):
            # Send data
            self.send(works[workload_id])
            # print('2. Send workload id %d' % workload_id)
            workload_id -= 1

            # Recv data
            data_to_merge = self.recv()
            # print('2. Recv workload id %d' % workload_id)
            assert len(data_to_merge) == len(works[workload_id])

            works[workload_id] = data_to_merge

        new_grads_0 = []
        for workload in works:
            new_grads_0 += workload
        grads[0] = new_grads_0
        return grads

    def merge_grad_to_optimizer(self, optimizer, grads):
        for index_p_g, p_g in enumerate(optimizer.param_groups):
            for index_p, p in enumerate(p_g['params']):
                optimizer.param_groups[index_p_g]['params'][index_p].grad = \
                    grads[index_p_g][index_p]

    # Train
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer = self.optimizer
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            grads = hooker.get_grad_from_optimizer(optimizer, self.world_size)
            grads = self.ring_allreduce(grads)
            # print('Grads length', len(grads[0]))
            self.merge_grad_to_optimizer(optimizer, grads)

            optimizer.step()
            # self.print_params(self.net)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(self, epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
