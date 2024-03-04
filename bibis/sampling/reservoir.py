
import heapq 
import random 
import numpy as np 

class UniformSampler:
    def __init__(self, size=1, seed=777):
        self.arr = []
        self.count = 0
        self.capacity = size
        self.rng = random.Random(seed)

    def get(self):
        return self.arr

    def add(self, item):
        self.count += 1
        
        if self.count <= self.capacity:
            self.arr.append(item)
        else:
            j = self.rng.randrange(0, self.count)
            if j < self.capacity:
                self.arr[j] = item
        return 

    def __repr__(self):
        return str(self.get())

class WeightSampler:
    # https://github.com/minddrummer/weightreservoir
    def __init__(self, size: int =1, seed=777):
        assert size >= 1
        self.heap = []
        self.count = 0
        self.capacity = size
        self.rng = random.Random(seed)

    def get(self):
        return [item[1] for item in self.heap]

    def add(self, item: int, weight: float):
        if weight<=0: return

        self.count += 1
        pair = (self.rng.random() ** (1.0/weight), item)
        if self.count < self.capacity:
            self.heap.append(pair)
        elif self.count == self.capacity:
            self.heap.append(pair)
            heapq.heapify(self.heap)
        else:
            if pair[0]>self.heap[0][0]:
                _ = heapq.heapreplace(self.heap, pair)
        return 

    def __repr__(self):
        return str(self.get())
    
class PredefinedSizeUniformSelector:
    def __init__(self, sample_size: int, total_size: int, seed: int=777):
        self.count = 0
        self.sample_size = sample_size
        self.total_size = total_size
        rng = np.random.default_rng(seed=seed)
        
        self.selected_ids = np.sort(
            rng.choice(total_size, 
                       size=sample_size, 
                       replace=False))
        self.cur_pos = 0

    def add(self, _):
        if self.count >= self.total_size:
            raise Exception(f"Observed more entries, than total size {self.total_size}")
        selected = False
        if self.cur_pos < self.sample_size and\
          self.count == self.selected_ids[self.cur_pos]:
            self.cur_pos += 1
            selected = True
        
        self.count += 1
        return selected
    
class PredefinedIndicesSelector:
    def __init__(self, indices):
        self.count = 0
        self.sample_size = len(indices)
        self.selected_ids = indices
        self.cur_pos = 0

    def add(self, _):
        selected = False
        if self.cur_pos < self.sample_size and\
                self.count == self.selected_ids[self.cur_pos]:
            self.cur_pos += 1
            selected = True
        self.count += 1
        return selected
    
class AllSelector:
    '''
    no sampling, just collect seen data
    '''
    def __init__(self) -> None:
        pass
    
    def add(self, item):
        return True