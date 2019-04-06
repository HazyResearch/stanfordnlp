from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import scipy
import scipy.sparse.csgraph as csg
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import math
from io import open
import unicodedata
import string
import re
import random
import json
from collections import defaultdict
# import utils.load_dist as ld

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Distortion calculations
def hyp_dist_origin(x):
    return torch.log(torch.div(1+torch.norm(x),1-torch.norm(x)))

def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

def _correct(x, eps=1e-1):
        current_norms = torch.norm(x,2,x.dim() - 1)
        mask_idx      = current_norms < 1./(1+eps)
        modified      = 1./((1+eps)*current_norms)
        modified[mask_idx] = 1.0
        return modified.unsqueeze(-1)

def dist_h(u,v):
    u = u * _correct(u)
    v = v * _correct(v)
    z  = 2*torch.norm(u-v,2)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2)**2)*(1-torch.norm(v,2)**2)))
    return acosh(uu)

def dist_e(u, v):
    return torch.norm(u-v, 2)

def dist_eb(u, v):
    return torch.norm(u-v, 2)

def dist_p(u,v):
    z  = 2*torch.norm(u-v,2)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2)**2)*(1-torch.norm(v,2)**2)))
    machine_eps = np.finfo(uu.data.detach().cpu().numpy().dtype).eps  # problem with cuda tensor
    return acosh(torch.clamp(uu, min=1+machine_eps))

def dist_pb(u,v):
    #print("u = ", u, " v = ", v)
    z  = 2*torch.norm(u-v,2, dim=1)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2, dim=1)**2)*(1-torch.norm(v,2, dim=1)**2)))
    machine_eps = np.finfo(uu.data.detach().cpu().numpy().dtype).eps  # problem with cuda tensor
    #print("distance was ", acosh(torch.clamp(uu, min=1+machine_eps)))
    print("THIS me = ", machine_eps)
    return acosh(torch.clamp(uu, min=1+machine_eps))

def distance_matrix_euclidean(input):
    row_n = input.shape[0]
    mp1 = torch.stack([input]*row_n)
    mp2 = torch.stack([input]*row_n).transpose(0,1)
    dist_mat = torch.sum((mp1-mp2)**2,2).squeeze()
    return dist_mat

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def distance_matrix_hyperbolic(input, sampled_rows, scale):
    row_n = input.shape[0]
    dist_mat = torch.zeros(len(sampled_rows), row_n, device=device)
    idx = 0
    for row in sampled_rows:
        for i in range(row_n):
            #if i != row:
            dist_mat[idx, i] = dist_p(input[row,:], input[i,:])*scale
        idx += 1
    #print("Distance matrix", dist_mat)
    #print()
    return dist_mat

def distance_matrix_hyperbolic_batch_old(input, sampled_rows, scale):
    #print("were computing the matrix with sampled_rows = ")
    #print(sampled_rows)
    batch_size = input.shape[0]
    row_n = input.shape[1]
    dist_mat = torch.zeros(batch_size, len(sampled_rows), row_n, device=device)
    # num_cores = multiprocessing.cpu_count()
    # dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    idx = 0
    for row in sampled_rows:
        for i in range(row_n):
            #if i != row:
            dist_mat[:,idx, i] = dist_pb(input[:,row,:], input[:,i,:])*scale
        idx += 1

    return dist_mat

def distance_matrix_hyperbolic_batch(input, sampled_rows, scale):
    batch_size = input.shape[0]
    row_n = input.shape[1]

    u = torch.stack([input]*row_n).transpose(0,1)
    v = torch.stack([input]*row_n).transpose(0,1).transpose(1,2)

    nrms = torch.norm(input, 2, 2)
    pr = torch.ones(batch_size, row_n).cuda() - nrms ** 2
    den = pr[:, :, None] @ pr[:, None, :]
    num = 2 * torch.sum((u-v)**2,3).squeeze() if row_n > 1 else 2 * torch.sum((u-v)**2,3)

    dist_mat = torch.ones(batch_size, row_n, row_n).cuda() + torch.div(num, den) * scale

    machine_eps = np.finfo(dist_mat.data.detach().cpu().numpy().dtype).eps  # problem with cuda tensor
    dist_mat = acosh(torch.clamp(dist_mat, min=1+machine_eps))

    return dist_mat

def distance_matrix_euclidean_batch(input, sampled_rows, scale):
    #print("were computing the matrix with sampled_rows = ")
    #print(sampled_rows)
    batch_size = input.shape[0]
    row_n = input.shape[1]
    dist_mat = torch.zeros(batch_size, len(sampled_rows), row_n, device=device)
    # num_cores = multiprocessing.cpu_count()
    # dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    idx = 0
    for b in range(batch_size):
        dist_mat[b,:,:] = distance_matrix_euclidean(input[b,:,:])
    #print("Distance matrix", dist_mat)
    return dist_mat

def entry_is_good(h, h_rec): return (not torch.isnan(h_rec)) and (not torch.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec):
    avg = abs(h_rec - h)/h
    avg += abs(h - h_rec)/h_rec
    avg /= 2

    return avg

def distortion_row(H1, H2, n, row):
    avg, good = 0, 0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            #if H1[i] <= 4:
            if True:  
                _avg = 1.0 / H1[i] * distortion_entry(H1[i], H2[i])
                #_avg = distortion_entry(H1[i], H2[i])
                good        += 1
                avg         += _avg
    if good > 0:
        avg /= good 
    else:
        avg, good = torch.tensor(0., device=device, requires_grad=True), torch.tensor(0., device=device, requires_grad=True)
    # print("Number of good entries", good)
    return (avg, good)


def distortion(H1, H2, n, sampled_rows, jobs=16):
    i = 0
    # print("h1", H1.shape)
    # print("h2", H2.shape)
    dists = torch.zeros(len(sampled_rows))
    for row in sampled_rows:
        dists[i] = distortion_row(H1[row,:], H2[i,:], n, row)[0]
        i += 1

    avg = dists.sum() / len(sampled_rows)
    return avg

def distortion_batch(H1, H2, n, sampled_rows):
    t = time.time()
    batch_size = H1.shape[0]
    diag_mask = torch.eye(n)
    diag_mask = diag_mask.unsqueeze(0)
    diag_mask = diag_mask.expand(batch_size, n, n).cuda()
    off_diag  = torch.ones(batch_size, n, n).cuda() - diag_mask
    
    os = torch.zeros(batch_size, n, n).cuda()
    ns = torch.ones(batch_size, n, n).cuda()
    H1m = torch.where(H1 > 0, ns, os).cuda()
    H2m = torch.where(H2 > 0, ns, os).cuda()
       
    good1 = torch.clamp(H1m.sum(), min=1)
    good2 = torch.clamp(H2m.sum(), min=1)

    # these have 1's on the diagonals. Also avoid having to divide by 0:
    H1_masked = H1 * off_diag + diag_mask + torch.ones(batch_size, n, n).cuda()*0.00001
    H2_masked = H2 * off_diag + diag_mask + torch.ones(batch_size, n, n).cuda()*0.00001

    dist1 = torch.div(torch.abs(H1_masked - H2_masked), H2_masked)
    dist2 = torch.div(torch.abs(H2_masked - H1_masked), H1_masked)

    H1_focus = ns  / (torch.clamp(H1_masked, min=1))

    l = ((dist1*H2m*H1_focus)).sum()/good1 + ((dist2*H1m*H1_focus)).sum()/good2
    #print("time to compute the loss = ", time.time()-t)
    return l


def distortion_batch_old(H1, H2, n, sampled_rows, graph, mapped_vectors, jobs=16):
    #print("First one\n")
    #print(H1)
    #print("Second one\n")
    #print(H2)

    # dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    # print(H1.shape) #target
    # print(H2.shape) #recovered
    batch_size = H1.shape[0]
    dists = torch.zeros(batch_size, len(sampled_rows))
    dists_orig = torch.zeros(batch_size)

    for b in range(batch_size):
        # let's add a term that captures how far we are in terms of getting the right guy in
        g_nodes = list(graph[b].nodes())
        root = g_nodes[0]
        '''
        print("root = ", root)
        print("location = ", mapped_vectors[b,root,:])
        print("Root norm = ", np.linalg.norm(mapped_vectors[b,root,:].detach().cpu().numpy()))
        print("Other norms = ")
        for i in range(n):
            print(np.linalg.norm(mapped_vectors[b,i,:].detach().cpu().numpy()))
        print()
        '''

        dists_orig[b] = hyp_dist_origin(mapped_vectors[b,root,:])
        i=0
        for row in sampled_rows:
            '''
            print("on row ", row) 
            print()
            print("true")
            print(H1[b,row,:])
            print("ours")
            print(H2[b,i,:])
            print()
            '''
            dists[b,i] = distortion_row(H1[b,row,:], H2[b,i,:], n, row)[0]
            i += 1

    #to_stack = [tup[0] for tup in dists]
    #avg = torch.stack(to_stack).sum() / len(sampled_rows)
    avg = dists.sum(dim=1)/len(sampled_rows)
    #print(" we added ", dists_orig)
    #print(" the normal is ", avg.sum())

    tot = (dists_orig.sum() * 1.0 + avg.sum())/batch_size
    return tot

def frac_distortion_row(H):
    return torch.fmod(H, 1).sum()

def frac_distortion(H, sampled_rows):
    frac_dists = torch.zeros(len(sampled_rows))

    for i in range(len(sampled_rows)):
        frac_dists[i] = frac_distortion_row(H[i,:])

    return frac_dists.sum() / len(sampled_rows)


def load_graph(file_name, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            if len(tokens) > 2:
                w = float(tokens[2])
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u,v)
    return G


def compute_row(i, adj_mat): 
    return csg.dijkstra(adj_mat, indices=[i], unweighted=True, directed=False)

def get_dist_mat(G):
    n = G.order()
    adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
    t = time.time()
    
    num_cores = multiprocessing.cpu_count()
    dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    return dist_mat

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def compare_mst(G, hrec):
    mst = csg.minimum_spanning_tree(hrec)
    G_rec = nx.from_scipy_sparse_matrix(mst)
    found = 0
    for edge in G_rec.edges():
        if edge in G.edges(): found+= 1

    acc = found / len(list(G.edges()))
    return acc

def compare_mst_batch(target_batch, hrec_batch):

    batch_size = hrec_batch.shape[0]
    batch_acc = 0
    for i in range(batch_size):
        hrec = hrec_batch[i,:,:]
        target = target_batch[i,:,:]
        mst = csg.minimum_spanning_tree(hrec)
        G_rec = nx.from_scipy_sparse_matrix(mst)
        mst_target = csg.minimum_spanning_tree(target)
        G = nx.from_scipy_sparse_matrix(mst_target)
        found = 1
        for edge in G_rec.edges():
            if edge in G.edges(): found+= 1

        acc = found / (len(list(G.edges()))+1)
        batch_acc += acc
    return batch_acc/batch_size



def predict_batch(target_batch, hrec_batch, sentlens):

    batch_size = hrec_batch.shape[0]
    node_system = 0
    node_gold = 0
    correct_heads = 0
    batch_acc = 0
    f1_total = 0
    for i in range(batch_size):
        ind = sentlens[i]
        hrec = hrec_batch[i,:ind,:ind]
        target = target_batch[i,:ind,:ind]
        mst = csg.minimum_spanning_tree(hrec)
        G_rec = nx.from_scipy_sparse_matrix(mst)
        mst_target = csg.minimum_spanning_tree(target)
        G = nx.from_scipy_sparse_matrix(mst_target)
        node_system += len(list(G_rec.nodes()))
        node_gold += len(list(G.nodes()))
        found = 1 #counting mst root to placeholder root node.
        for edge in G_rec.edges():
            if edge in G.edges(): found+= 1
        correct_heads += found
        acc = found / (len(list(G.edges()))+1)
        recall = acc
        precision = found / (len(list(G_rec.edges()))+1)
        f1 = 2*precision*recall/(precision+recall)
        batch_acc += acc
        f1_total += f1
    batch_acc /= batch_size
    return batch_acc, f1_total, correct_heads, node_system, node_gold


def unroll(node, G):
    if len(node.children) != 0:
        for child in node.children:
            G.add_edge(node.token['id'], child.token['id'])
            unroll(child, G)
    return G


def compute_row(i, adj_mat): 
    return csg.dijkstra(adj_mat, indices=[i], unweighted=True, directed=False)
    
def save_dist_mat(G, file):
    n = G.order()
    print("Number of nodes is ", n)
    adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
    t = time.time()
    
    num_cores = multiprocessing.cpu_count()
    dist_mat = Parallel(n_jobs=20)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    print("Time elapsed = ", time.time()-t)
    pickle.dump(dist_mat, open(file,"wb"))

def load_dist_mat(file):
    return pickle.load(open(file,"rb"))

def unwrap(x):
    """ Extract the numbers from (sequences of) pytorch tensors """
    if isinstance(x, list) : return [unwrap(u) for u in x]
    if isinstance(x, tuple): return tuple([unwrap(u) for u in list(x)])
    return x.detach().cpu().numpy()


def get_dist_mat(G, parallelize=False):
    n = G.order()
    adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
    t = time.time()
    num_cores = multiprocessing.cpu_count() if parallelize else 1
    dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    return dist_mat



def get_heads_batch(hrec_batch, sentlens, roots):

    batch_size = hrec_batch.shape[0]
    preds = []
    #placeholder
    rel = 'obj'

    for b in range(batch_size):
        hrec = hrec_batch[b,:,:]
        size = sentlens[b]
        root = roots[b]
        hrec = hrec[:size,:size]
        mst = csg.minimum_spanning_tree(hrec)
        G = nx.from_scipy_sparse_matrix(mst)
        seq = []
        head_dict = {}
        head_dict[root] = 'root'

        def find_heads(root, G, head_dict):
            neighbor_list = [n for n in G.neighbors(root)]
            if len(neighbor_list) != 0:
                for neighbor in neighbor_list:
                    if neighbor not in head_dict.keys():         
                        head_dict[neighbor] = root
                        find_heads(neighbor, G, head_dict)

            return head_dict
        
        head_dict = find_heads(root, G, head_dict)
        # print("sent len", size)
        # print("head dict", len(head_dict.keys()))
        keylist = head_dict.keys()
        keylist = sorted(keylist)
        for key in keylist:
            if head_dict[key] == 'root':
                seq.append(['0', 'root'])
            else:
                seq.append([str(head_dict[key]+1), rel])


        preds += [seq]


    return preds



