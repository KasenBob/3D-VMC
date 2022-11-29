import queue
import random
from collections import namedtuple, deque
import numpy as np
from random import sample, choice, randrange, uniform

Transition = namedtuple('Transition', ['images', 'aug_images', 'volume', 'taxonomy_name', 'sample_name'])


class taxonomy_info():
    def __init__(self):
        super().__init__()
        self.dict = {}
        self.sum = 0
    
    def add(self, tax):
        if tax in self.dict:
            self.dict[tax] += 1
            self.sum += 1
        else:
            self.dict[tax] = 1
    
    def get_prob(self, tax):
        return float(self.dict[tax] / self.sum)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayMemory_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity=1000, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, *args):
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        idxs = []
        segment = self.tree.total() / batch_size
        sample_datas = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = uniform(a, b)
            idx, p, data = self.tree.get(s)

            sample_datas.append(data)
            idxs.append(idx)
        return idxs, sample_datas

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))

        d = {}
        d = d.fromkeys(idxs)
        idxs = list(d.keys())

        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries