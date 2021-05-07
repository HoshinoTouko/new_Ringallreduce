import time

from utils import progress_bar

import os
import torch

import hooker
import pickle
import copy
import time
import random
import numpy as np
from numpy import mean, std
from queue import Queue


class Trainer:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    ENDFLAG = b'BINARY_END'
    RECV_BUFFER = 150000 * 1024

    def __init__(self, net, rank, world_size, trainloader, testloader, optimizer, criterion, socket_send, socket_recv,dl):
        self.net = net
        self.rank = rank
        self.world_size = world_size
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.socket_send = socket_send
        self.socket_recv = socket_recv
        self.best_acc = 0
        self.sticky_cache = Queue()
        self.dl = dl
        self.tmp_delay = []
        self.itera = 0

        self.delay = []
        self.k_list = np.zeros(10000)
        # self.k_list = {'kvalue': np.zeros(
            # 10000), 'kinc': 0.005, 'kdec': 0.005, 'kmin': 0.005, 'kmax': 0.15}
        # self.para = {'slack': 1.25, 'dec_factor': 0.8}

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
        if len(data_to_return) == 2:
            self.tmp_delay.append(int(round(time.time() * 1000)) - data_to_return['timestamp'])
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

    def AD2(self,i):
        min_delay = min(self.delay[:i+1])
        avg_delay = np.mean(self.delay[:i+1])
        relative_delay = np.std(self.delay[:i+1],ddof=1)
        cur_delay = self.delay[i]

        kinc = 0.005
        kdec = 0.005
        kmin = 0.005
        kmax = 0.15
        slack = 1.15
        dec_factor = 0.8
        if cur_delay < slack*min_delay:
            print("**enter 1")
            self.k_list[i+1] = self.k_list[i] + kinc
        else:
            if cur_delay > slack*avg_delay:
                print("**enter 2")
                tmp = (cur_delay - avg_delay)/(cur_delay - slack*min_delay)
                self.k_list[i+1] = self.k_list[i]*(1-dec_factor*tmp)
            else:
                tmp = relative_delay/min_delay
                if tmp < 0 :
                    print("**enter 3")  
                    self.k_list[i+1] = self.k_list[i]+kinc
                else:
                    print("**enter 4")
                    self.k_list[i+1] = self.k_list[i]*(1-dec_factor*tmp)

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

    def ring_allreduce_loss(self, grads):
        works = self.cut(grads[0])

        workload_id = self.rank
        self.tmp_delay = []
        
        # for i in range(1, len(works[workload_id])):
        #     print(">>>>>>dim 0:")
        #     print(works[workload_id][i][0])
        #     lens = len(works[workload_id][i])
        #     values, indices = works[workload_id][i].topk(
        #         int(lens*0.98), dim=0, largest=True, sorted=True)
        #     print(values)
        #     print(indices)
        #     # for j in range(lens):
        #     #     if j not in indices:
        #     #         works[workload_id][i][j] = 0
        # exit()
        for _ in range(self.world_size - 1):
            # top k
            # for i in range(1, len(works[workload_id])):
            #     # print(">>>>>>dim 0:")
            #     # print(works[workload_id][i][0])
            #     lens = len(works[workload_id][i])
            #     values, indices = works[workload_id][i].topk(
            #         int(lens*0.98), dim=0, largest=True, sorted=True)
            #     for j in range(lens):
            #         if j not in indices:
            #             works[workload_id][i][j] = 0

            # random k
            for i in range(1, len(works[workload_id])):
                # print(">>>>>>dim 0:")
                # print(works[workload_id][i][0])
                lens = len(works[workload_id][i])
                # values, indices = works[workload_id][i].topk(
                #     int(lens*0.98), dim=0, largest=True, sorted=True)
                kk = random.sample(range(0, lens),int(lens*0.98))
                # print(kk)
                for j in range(lens):
                    if j not in kk:
                        works[workload_id][i][j] = 0

            # Send data
            data = {'grad':works[workload_id], 'timestamp':int(round(time.time() * 1000))}
            self.send(data)
            # print('1. Send workload id %d' % workload_id)
            workload_id -= 1

            # Recv data
            data_to_merge = self.recv()
            # print('1. Recv workload id %d' % workload_id)
            assert len(data_to_merge['grad']) == len(works[workload_id])
            # self.tmp_delay.append(int(round(time.time() * 1000)) - data_to_merge['timestamp'])

            for index, item in enumerate(data_to_merge['grad']):
                works[workload_id][index] += item

        if self.rank == 0:
            workload_id += self.world_size

        for _ in range(self.world_size - 1):
            # Send data
            data = {'grad':works[workload_id], 'timestamp':int(round(time.time() * 1000))}
            self.send(data)
            # print('2. Send workload id %d' % workload_id)
            workload_id -= 1

            # Recv data
            data_to_merge = self.recv()
            # print('2. Recv workload id %d' % workload_id)
            assert len(data_to_merge['grad']) == len(works[workload_id])

            works[workload_id] = data_to_merge['grad']

        new_grads_0 = []
        for workload in works:
            new_grads_0 += workload
        grads[0] = new_grads_0
        # print('len of tmp_delay: '+str(len(self.tmp_delay)))
        self.delay.append(mean(self.tmp_delay))
        self.AD2(self.itera)
        self.itera += 1
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
            if self.dl == 1:
                grads = self.ring_allreduce_loss(grads)
            else:
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
        with open('./output/train_output_' + str(self.rank) + '.txt', 'a') as f:
            f.write(str(epoch) + ' ' + str(train_loss) + ' ' +
                    str(100 * correct / total) + ' '+str(time.time()) + '\n')

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(
                    self.DEVICE), targets.to(self.DEVICE)
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
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc

        with open('./output/test_output_' + str(self.rank) + '.txt', 'a') as f:
            f.write(str(epoch) + ' ' + str(test_loss) + ' ' +
                    str(acc) + ' '+str(time.time())+'\n')
