import time

import compress
from utils import progress_bar

import os
import sys
import torch

import hooker
import pickle
import copy
import time
import random
import numpy as np
from numpy import mean, std
from queue import Queue

from scipy.sparse import csr_matrix
from scipy import sparse

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

    def cut(self, grads):
        workload = int(len(grads) / self.world_size) + 1
        return [
            grads[workload * offset:workload * (offset + 1)]
            for offset in range(self.world_size)
        ]

    def cut_with_compress(self, grads):
        workload = int(len(grads) / self.world_size) + 1
        return [
            [
                compress.compress(_grads.cpu())
                for _grads in grads[workload * offset:workload * (offset + 1)]
            ]
            for offset in range(self.world_size)
        ]

    def ring_allreduce_with_loss(self, grads):
        works = self.cut(grads[0])
        works_to_send = self.cut_with_compress(grads[0])

        workload_id = self.rank
        for _ in range(self.world_size - 1):
            # Send data
            self.send(works_to_send[workload_id])
            # print('1. Send workload id %d' % workload_id)
            workload_id -= 1

            # Recv data
            # data_to_merge = self.recv()
            data_to_merge = []
            recved = self.recv()
            for shape, compressed_data, zero_mask_bitmap in recved:
                data_to_merge.append(compress.uncompress(shape, compressed_data, zero_mask_bitmap).to(self.DEVICE))
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

    def compress(self,value):
        s_value = []
        shape = []
        bitmap = []
        print("origin lens:"+str(len(value)))
        lens = len(value)
        for i in range(0, lens):
            # random k 0.2
            random_data = torch.rand(value[i].size())
            value[i][random_data<0.2] = 0

            np_value = value[i].cpu().numpy()
            shape.append(np_value.shape)

            flattened_value = np_value.flatten()
            zero_mask = np.zeros(flattened_value.shape, dtype=np.int)
            zero_mask[flattened_value == 0] = 1

            vals_position = np.where(zero_mask == 0)
            zeros_position = np.where(zero_mask == 1)

            zero_mask_bitmap = np.packbits(zero_mask)
            # print(sys.getsizeof(pickle.dumps(zero_mask_bitmap)))

            compressed_data = flattened_value[vals_position]
            s_value.append(compressed_data)
            bitmap.append(zero_mask_bitmap)
            print("origin :dim of zero: "+str((zero_mask == 0).shape)+" dim of cmp data: "+str(compressed_data.shape))
            # print(len(flattened_value), len(compressed_data))
        return [shape, s_value, bitmap,lens]

    def uncompress(self,value):
        d_value = []
        for i in range(0, value[-1]):
            total_length = np.prod(value[0][i])
            uncompressed_data = np.zeros([total_length])
            zero_mask = np.unpackbits(value[2][i])

            print("dim of zero: "+str((zero_mask == 0).shape)+" dim of uncmp data: "+str(uncompressed_data.shape))
            uncompressed_data[zero_mask == 0] = value[1][i]
            # print(len(uncompressed_data[zero_mask == 0]), len(value[1][i]))
            uncompressed_data = uncompressed_data.reshape(value[0][i])
            
            d_value.append(torch.from_numpy(uncompressed_data).to(self.DEVICE))

        return d_value

    def ring_allreduce_loss(self, grads):
        works = self.cut(grads[0])

        workload_id = self.rank
        self.tmp_delay = []

        for _ in range(self.world_size - 1):

            # Send data
            # print("send: "+str(works[workload_id]))
            data = {'grad':self.compress(works[workload_id]), 'timestamp':int(round(time.time() * 1000))}
            # print(sys.getsizeof(pickle.dumps(works[workload_id])))
            # print(sys.getsizeof(self.compress(works[workload_id])))
            self.send(data)
            # print('1. Send workload id %d' % workload_id)
            workload_id -= 1

            # Recv data
            recv_data = self.recv()
            data_to_merge = self.uncompress(recv_data['grad'])
            # print(data_to_merge)

            # print('1. Recv workload id %d' % workload_id)
            assert len(data_to_merge) == len(works[workload_id])

            # print("uncompress: ")
            # print(data_to_merge)
            for index, item in enumerate(data_to_merge):
                works[workload_id][index] += item
            # print(self.value)
            # print(data_to_merge)
            # print(self.value[1].equal(data_to_merge[1]))
            exit()

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
                grads = self.ring_allreduce_with_loss(grads)
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
