import numpy as np
import scipy


def create_unit_cardinality(n):
    # tr = np.array(range(n))[np.newaxis].T
    return np.array(range(n))


def create_singleton_partition(n):
    l = []
    for i in list(range(n)):
        l.append([i])
    return np.array(l)

    # verticies


def ver(graph):
    return graph.shape[0]

def weighted_sum(xs, weights):
    return sum(weights[s] for s in xs if xs.size)

def partition_graph(admat, cardinality, partition): #edge_weights
    # self.check_adjacent_matrix(admat)
    n = ver(admat)
    # self.check_cardinality(n, cardinality)
    # self.check_partition(n, partition)
    g_size = np.zeros(len(partition))
    membership = np.zeros(n)
    for (i, com) in enumerate(partition):
        g_size[i] = weighted_sum(com, cardinality)
        membership[com] = i
    return [admat, cardinality, partition, g_size, membership]  #edge_weights


# number of communities
def nc(part_graph):
    return len(part_graph[2])


# neibours of chosen vert
"""def neighbors(graph, ver):
    A = graph[5]
    a1 = scipy.csr_matrix(A)
    a1.indeces[a1.indptr[ver]:a1.indptr[ver + 1] - 1]
    return a1"""
def neighbors(part_graph,ver):
    A = part_graph[0]
    l=[]
    for i in range(A.shape[0]):
        if A[i,ver] == 1:
            l.append(i)
        if A[ver,i] == 1 and i not in l:
            l.append(i)
    return l
    # communities connected with ver
    # def con_comm(self,graph,ver):
    #   pass
    #  return

    # generstors of default parametrs

def drop_empty_com(part_graph):
    empty = np.array([])

    for (i, com) in enumerate(part_graph[2]):
        if not com.size:
            empty.append(i)

    if empty in part_graph[2]:
        np.delete(part_graph[2], empty)
    if empty in part_graph[3]:
        np.delete(part_graph[3], empty)
    for (i, community) in enumerate(part_graph[2]):
        part_graph[4][community] = i
    return part_graph

def resert_partition(part_graph, partition):
    # check_partition
    part_graph[2] = []
    part_graph[3] = []
    for (i, com) in enumerate(partition):
        part_graph[2].append(com)
        part_graph[3].append(weighted_sum(com, part_graph[1]))
        part_graph[4][com] = i
    return part_graph



