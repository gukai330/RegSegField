import numpy as np

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

def global_object_id(N, M):
    total_objects = sum(N)
    uf = UnionFind(total_objects)

    current_id = 0
    for i in range(len(N)):
        for j in range(i + 1, len(N)):
            for x in range(N[i]):
                for y in range(N[j]):
                    if M[i][j][x][y] >= 1:
                        # 更新对应物体的全局编号
                        uf.union(current_id + x, sum(N[:j]) + y)
        current_id += N[i]

    
    global_id_mapping = [uf.find(i) for i in range(total_objects)]
    # make split mapping
    object_id_cumsum = np.cumsum(N)

    splited_global_id_mapping = np.split(global_id_mapping, object_id_cumsum)

    return splited_global_id_mapping[:len(N)]