import numpy as np
import Graph
#from goto import with_goto
from scipy import special
import math
import random
import sys
sys.setrecursionlimit(5000)
from scipy.sparse import csc_matrix



def leiden(adMat, resolution = 1.0, randomness = 0.01 ):

    partition = Graph.create_singleton_partition(adMat.shape[0])
    cardinality = Graph.create_unit_cardinality(adMat.shape[0])

    graph = Graph.partition_graph(adMat,cardinality= cardinality,partition= partition)

    return leiden1(graph,resolution,randomness)

def leiden1(graph,gamma,teta):
    stack = []
    refined = refine_partition(graph, gamma, teta)

    while Graph.nc(refined) != Graph.ver(refined[0]):
        move_nodes_fast(graph,gamma)
        nv1 = Graph.ver(graph)
        nc1 = Graph.nc(graph)
        if nc1 != nv1:
            refined = refine_partition(graph, gamma,teta)
            stack.apend(refined[2])
            graph1 = aggregate_graph(refined)
            partition = [[] for _ in list(range(Graph.nc(graph))) ]
            for (i,com) in enumerate(refined[2]):
                u= com[0]
                j = graph[4][u]
                partition[int(j)].append(i)
            Graph.resert_partition(graph1,partition)
            graph = graph1
            continue
    stack.append(graph[2])
    quality = H(graph,gamma)
    part = flatten(stack)
    return [quality,part]



def move_nodes_fast(graph,gamma):
    n = Graph.ver(graph[0])
    l = list(range(n))
    random.shuffle(l)
    queue = l
    queued = list(range(n))
    total_weights = np.zeros(Graph.nc(graph))
    while len(queue)>0:
        u = queue.pop(0)
        queued.remove(u)

        # compute total edge weights for each community
        connected = []
        for v in Graph.neighbors(graph, u):
            i = graph[4][v]
            if total_weights[int(i)] == 0:
                connected.append(i)
            total_weights[int(i)] += graph[0][u, v]

        # find the best community to which `u` belongs
        c_u = graph[1][u]
        weight_u = graph[0][u, u]
        src = dst = graph[4][u]
        weight_src = total_weights[int(src)]
        size_src = graph[3][int(src)]
        maxgain = 0.0
        for i in connected:
            i == src
            gain = total_weights[int(i)] + weight_u - weight_src - gamma * (graph[3][int(i)] - size_src + c_u) * c_u
            if gain > maxgain:
                dst = i
                maxgain = gain

            total_weights[int(i)] = 0

        total_weights[int(src)] = 0

        if src != dst:
            # move `u` to the best community and add its neighbors to the queue if needed
            move_nodes(graph, gamma)
            for v in Graph.neighbors(graph,u):
                if graph[4][v] != graph[4][u] and v not in queued:
                    queue.append(v)
                    queued.append(v)

    return  Graph.drop_empty_com(graph)

def move_nodes(graph, gamma):
    H_old = H(graph,gamma)
    nodes =list(range(Graph.ver(graph[0])))
    total_weights = np.zeros(Graph.nc(graph))
    H_new = H_old+1
    while H_new >=  H_old:
        random.shuffle(nodes)
        for u in nodes:
            #compute total edge weights for each community
            connected = []
            for v in Graph.neighbors(graph, u):
                i = graph[4][v]
                if total_weights[int(i)] == 0:
                    connected.append(i)

                total_weights[int(i)] += graph[0][u, v]

                # find the best community to which `u` belongs
                c_u = graph[1][u]
                weight_u = graph[0][u, u]
                src = dst = graph[4][u]
                weight_src = total_weights[int(src)]
                size_src = graph[3][int(src)]
                maxgain = 0.0
                for i in connected:
                    i == src
                    gain = total_weights[int(i)] + weight_u - weight_src - gamma * (graph[3][int(i)] - size_src + c_u) * c_u
                    if gain > maxgain:
                        dst = i
                        maxgain = gain
                    total_weights[int(i)] = 0
                total_weights[int(src)] = 0

                if src != dst:

                    gr1 = move_nodes(graph,gamma)
        H_new = H(gr1,gamma)
        gr1 = []
    return Graph.drop_empty_com(graph)


def refine_partition(graph, gamma, teta):
    refined = Graph.partition_graph(graph[0],cardinality=graph[1],partition= graph[2])
    def is_well_connected1(u):
        i = graph[4][u]
        c = graph[1][u]
        threshold = gamma * c * (graph[3][int(i)] - c)
        x = 0
        for v in graph[2][int(i)]:
            v == u
            x += graph[0][u, v]
            if x >= threshold:  # return as early as possible
                return True
        return False

    def is_well_connected(u, i, between_weights):
        sz = refined[3][i]
        return between_weights[i] >= gamma * sz * (graph[3][graph[4][u]] - sz)

    def is_singleton(u):
        return len(refined[2][int(refined[4][u])])==1

    total_weights = np.zeros(Graph.nc(refined))
    between_weights = np.zeros(Graph.nc(refined))
    for sub in graph[2]:
        for u in sub:
            weight = 0
            for v in sub:
                if v != u :
                    weight +=refined[0][u,v]
            between_weights[int(refined[4][u])] = weight
        random.shuffle(sub)
        for u in sub:
            if not is_well_connected1(u) or not is_singleton(u):
                continue
            communities =[]
            for v in Graph.neighbors(refined,u):
                if v not in sub:
                    continue
                i = refined[4][v]
                if total_weights[int(i)]==0:
                    communities.append(i)
                total_weights[int(i)] +=refined[0][u,v]
            c_u = refined[1][u]
            weight_u = refined[0][u,v]
            scr = refined[4][u]
            weight_scr = total_weights[int(scr)]
            size_scr = refined[3][int(scr)]
            logprobs = []
            indexes=[]
            for i in communities:
                if i == scr or not is_well_connected(u,i,between_weights):
                    gain= total_weights[int(i)] + weight_u-weight_scr-gamma*(refined[3][int(i)]-size_scr+c_u)*c_u
                if gain>0:
                    logprobs.append( gain/teta)
                    indexes.append(i)
            total_weights[communities] = 0

            if  len(indexes)==0:
                continue

            probs = math.exp(logprobs-special.logsumexp(logprobs))
            dst = indexes[random.choice(probs)]

            move_nodes_fast(refined,gamma)
            for v in Graph.neighbors(refined,u):
                if v == u or v not in sub:
                    i = refined[4][v]
                    weight = refined[0][u,v]
                if  i==dst:
                    between_weights[int(scr)] -=weight
                    between_weights[int(dst)] -=weight
                elif i == scr:
                    between_weights[int(scr)] +=weight
                    between_weights[int(dst)] += weight
                else:
                    between_weights[int(scr)] -=weight
                    between_weights[int(dst)] +=weight
        for u in sub:
            between_weights[int(refined[4][u])] = 0
    return Graph.drop_empty_com(refined)

def sample(probs):
    r = random.random()
    p=0
    i=1
    while i <len(probs)-1:
        p+=probs[i]
        if p>r:
            return i
        i+=1
    return len(probs)-1

def aggregate_graph(graph):
    n = Graph.nc(graph)
    I = []
    J = []
    V = []
    cardinality = np.zeros(n)
    total_weights = np.zeros(n)
    #connected = []
    for (i,com) in enumerate(graph[2]):
        cardinality[i] = graph[3][i]
        connected = []
        for u in com:
            for v in Graph.neighbors(graph,u):
                j = graph[4][v]
                if i == j and u>v:
                    continue
                if total_weights[int(j)] ==0:
                    connected.append(j)
                total_weights[int(j)]+=graph[0][u,v]
        for j in connected:
            v = total_weights[int(j)]
            I.append(i)
            J.append(j)
            V.append(v)
            total_weights[int(j)] = 0
    return Graph.partition_graph(csc_matrix((V,(J,I)),shape=(n,n)).toarray(),cardinality=cardinality,partition=
                                 graph[2])

def H(graph,gamma):
    quality = 0
    for (i,com) in enumerate(graph[2]):
        for u in com:
            for v in com:
                if u<=v:
                    quality+=graph[0][u,v]
        n=graph[3][i]
        quality-=gamma* (n*(n-1)//2)
    return  quality

def flatten(stack):
    k=len(list(stack))-1
    res = stack[k]
    k-=1
    while k>=0:
        part = stack[k]
        for (i,indices) in enumerate(res):
            res[i] = part[i]
        k-=1
    sorted(res,key = len)
    return res