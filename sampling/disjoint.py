from dataclasses import dataclass
import numpy as np 

@dataclass
class DisjointSet:
    """
    GC-sampler based on DisjointSet
    """
    parent: np.ndarray
    power: np.ndarray
    left_brd: np.ndarray
    right_brd: np.ndarray
    is_taken: np.ndarray

    @classmethod
    def from_negative_gc(cls, neg_gc):
        N = neg_gc.shape[0]
        return cls.of_size(N)
    
    @classmethod
    def of_size(cls, N):
        parent = np.arange(N)
        power = np.ones(N)
        left_brd = np.arange(N)
        right_brd = left_brd + 1
        is_taken = np.zeros(N, dtype=np.bool8)
        return cls(parent, power, left_brd, right_brd, is_taken)

    def root(self, i):
        """
        get root of the cluster 
        """
        j = i
        while self.parent[j] != j:
            j = self.parent[j]
        while self.parent[i] != j: # path compression heuristic
            i, self.parent[i] =  self.parent[i], j
            
        return j 
    
    def join(self, i, j):
        """
        join two clusters and update their borders respectively
        """
        ri = self.root(i)
        rj = self.root(j)
        l = min(self.left_brd[ri], self.left_brd[rj])
        r = max(self.right_brd[ri], self.right_brd[rj])

        # append smaller dataset to larger 
        if self.power[rj] > self.power[ri]:
            self.parent[ri] = rj
            self.left_brd[rj] = l
            self.right_brd[rj] = r
            return rj
        elif self.power[ri] > self.power[rj]:
            self.parent[rj] = ri
            self.left_brd[ri] = l
            self.right_brd[ri] = r
            return ri
        else: # self.power[ri] == self.power[rj]:
            self.parent[ri] = rj
            self.left_brd[rj] = l
            self.right_brd[rj] = r
            self.power[rj] +=1
            return rj
    
    def left(self, i):
        """
        get left border of cluster, to which i corresponds 
        """
        ri = self.root(i)
        return self.left_brd[ri]
    
    def right(self, i):
        """
        get rigth border of cluster, to which i corresponds 
        """
        ri = self.root(i)
        return self.right_brd[ri]

    def take(self, i):
        self.is_taken[i] = True 
        p = self.join(i, i-1) # join with right-most left cluster 
        # now left border is valid    
        return p