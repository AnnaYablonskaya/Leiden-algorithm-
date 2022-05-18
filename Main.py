import sys
sys.setrecursionlimit(5000)
import numpy as np
import Graph
import Leiden
import pandas as pd
import ast
import igraph as ig
import leidenalg as la

if __name__ == '__main__':

    """gr = np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3]])
    weights =np.zeros((6,6))
    for [u,v] in gr:
        weights[u,v]  =1
        weights[v,u] = 1

    cardin = Graph.create_unit_cardinality(n=6)
    s_part = Graph.create_singleton_partition(n=6)
    v = Graph.ver(weights)
    part_Gr = Graph.partition_graph(weights,cardin,s_part)
    num_com = Graph.nc(part_Gr)
    neib = Graph.neighbors(part_Gr,0)
    dr_em  = Graph.drop_empty_com(part_Gr)
    res_part = Graph.resert_partition(part_Gr,s_part)
    sam = Leiden.sample(cardin)
    ar_gr = Leiden.aggregate_graph(part_Gr)
    hh = Leiden.H(ar_gr,0.04)
    #m_n = Leiden.move_nodes(ar_gr,0.04)
    #m_n_f = Leiden.move_nodes_fast(ar_gr,0.04)
    ref_gr = Leiden.refine_partition(ar_gr,0.04,0.05)
    l1 = Leiden.leiden1(ar_gr,0.04,0.05)
    l  =Leiden.leiden(weights)
    print(l) 
 
    """


