# Source: https://github.com/tracy-talent/curriculum/blob/ecaf850cb7932f23b5d7c0323e80a9f9a408bef6/Machine%20Learning/Dimension%20Reduction/src/ISOMAP.py

from numpy import *
from hw4.libs.metrics import _1NN
from queue import PriorityQueue
from os import path
import time


def loadData(filename):
    content = open(filename).readlines()
    # Split data and labels
    data = [list(map(float32, line.strip().split(",")[:-1])) for line in content]
    tag = [list(map(int, line.strip().split(",")[-1:])) for line in content]
    return mat(data), mat(tag)


def calc_distance(dataMat):
    dataSize = len(dataMat)
    Euc_distanceMat = zeros([dataSize, dataSize], float32)
    for i in range(dataSize):
        for j in range(dataSize):
            Euc_distanceMat[i][j] = linalg.norm(dataMat[i] - dataMat[j])
    return Euc_distanceMat


# Adjacency table edge
class edge(object):
    def __init__(self, cost, to):
        self.cost = cost  # 边权重
        self.to = to  # 入点

    def __lt__(self, other):
        return self.cost < other.cost


# dijkstra (Shortest path algorithm)
# @param{dist: Distance matrix, graph: Adjacency, src: Source}
def dijkstra(dist, graph, src):
    que = PriorityQueue()
    que.put(edge(0, src))
    while not que.empty():
        p = que.get()
        v = p.to
        if dist[src][v] < p.cost:
            continue
        for i in range(len(graph[v])):
            if dist[src][graph[v][i].to] > dist[src][v] + graph[v][i].cost:
                dist[src][graph[v][i].to] = dist[src][v] + graph[v][i].cost
                que.put(edge(dist[src][graph[v][i].to], graph[v][i].to))


# @param{dist:Distance matrix，dims: Dimensionality reduction / Number of Components}
# return：降维后的矩阵
def mds(dist, dims):
    dataSize = len(dist)
    if dims > dataSize:
        print('Dimension reduction dimension %d is greater than the dimension of the matrix to be reduced %d' % (dims, dist.shape()))
        return
    dist_i_dot_2 = zeros([dataSize], float32)
    dist_dot_j_2 = zeros([dataSize], float32)
    dist_dot_dot_2 = 0.0
    bMat = zeros([dataSize, dataSize], float32)
    for i in range(dataSize):
        for j in range(dataSize):
            dist_i_j_2 = square(dist[i][j])
            dist_i_dot_2[i] += dist_i_j_2
            dist_dot_j_2[j] += dist_i_j_2 / dataSize
            dist_dot_dot_2 += dist_i_j_2
        dist_i_dot_2[i] /= dataSize
    dist_dot_dot_2 /= square(dataSize)
    for i in range(dataSize):
        for j in range(dataSize):
            dist_i_j_2 = square(dist[i][j])
            bMat[i][j] = -0.5 * (dist_i_j_2 - dist_i_dot_2[i] - dist_dot_j_2[j] + dist_dot_dot_2)
    # Eigenvalues ​​and eigenvectors
    eigVals, eigVecs = linalg.eig(bMat)
    # Index for large eigenvalues
    eigVals_Idx = argpartition(eigVals, -dims)[:-(dims + 1):-1]
    # Constructing a Diagonal Matrix of Eigenvalues
    eigVals_Diag = diag(maximum(eigVals[eigVals_Idx], 0.0))
    return matmul(eigVecs[:, eigVals_Idx], sqrt(eigVals_Diag))


# param{dataMat: Image Dataset (n_data, n_dimension)，dims: Number of Components，KNN_K: Number of Neighbours}
# return：Dimensionality-reduced matrix
def isomap(dataMat, dims, KNN_K):
    set_printoptions(threshold=None)
    inf = float('inf')
    dataSize = len(dataMat)
    if KNN_K >= dataSize:
        # raise ValueError('KNN_K的值最大为数据个数 - 1:%d' % dataSize - 1)
        raise ValueError("The maximum value is the number of data = ", (dataSize-1))
    Euc_distanceMat = calc_distance(dataMat)
    # Setup KNN Connection diagram
    knn_distanceMat = ones([dataSize, dataSize], float32) * inf
    for i in range(dataSize):
        knn_disIdx = argpartition(Euc_distanceMat[i], KNN_K)[:KNN_K + 1]
        knn_distanceMat[i][knn_disIdx] = Euc_distanceMat[i][knn_disIdx]
        for j in knn_disIdx:
            knn_distanceMat[j][i] = knn_distanceMat[i][j]

    # Build adjacency list
    adjacencyTable = []
    for i in range(dataSize):
        edgelist = []
        for j in range(dataSize):
            if knn_distanceMat[i][j] != inf:
                edgelist.append(edge(knn_distanceMat[i][j], j))
        adjacencyTable.append(edgelist)

    # dijkstra: Find the shortest
    # dist: Store the shortest distance between any two points
    dist = ones([dataSize, dataSize], float32) * inf
    for i in range(dataSize):
        dist[i][i] = 0.0
        dijkstra(dist, adjacencyTable, i)
    return mds(dist, dims)

