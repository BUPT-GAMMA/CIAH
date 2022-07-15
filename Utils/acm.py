# -*- encoding: utf-8 -*-
import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from dgl.data.utils import download, get_download_dir, _get_dgl_url
# from pprint import pprint
from scipy import sparse
from scipy import io as sio

from Utils.multihot2index import multihot2index

"""
此ACM版本和赵建安的基本一致
'author': 7167, 'field': 60, 'hyperedge': 4025, 'paper': 4025
"""

def get_binary_mask(total_size, indices):
    mask = np.zeros(total_size)
    mask[indices] = 1
    return mask.astype(np.bool)

def load_acm_raw_hypergraph(onehot_for_nofeature=True, node_types="paf"):
    """
    This code & dataset is modified based on DGL library.
    Thanks to https://github.com/dmlc/dgl
    """
    url = 'dataset/ACM.mat'
    data_path = './Dataset/ACM/ACM.mat'
    # try:
    #     data = sio.loadmat(data_path)
    # except FileNotFoundError:
    download(_get_dgl_url(url), path=data_path, overwrite=False)
    data = sio.loadmat(data_path)

    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
    a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
    t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
    # c_selected = (p_vs_c[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected].T[l_selected].T
    p_vs_a = p_vs_a[p_selected].T[a_selected].T
    p_vs_t = p_vs_t[p_selected].T[t_selected].T
    p_vs_c = p_vs_c[p_selected]  # .T[c_selected].T  # 这里不改是为了后面label正确

    hg = dgl.heterograph({
        ('paper', 'pe', 'hyperedge'): np.eye(p_vs_l.shape[0], dtype=int).nonzero(),
        ('hyperedge', 'ep', 'paper'): np.eye(p_vs_l.shape[0], dtype=int).transpose().nonzero(),
        # P和超边等价
        ('author', 'ae', 'hyperedge'): p_vs_a.transpose().nonzero(),
        ('hyperedge', 'ea', 'author'): p_vs_a.nonzero(),
        ('field', 'fe', 'hyperedge'): p_vs_l.transpose().nonzero(),
        ('hyperedge', 'ef', 'field'): p_vs_l.nonzero(),
    })
    E4N_adjs = [
        hg.adj(etype='ep', scipy_fmt="coo", transpose=True),
        hg.adj(etype='ea', scipy_fmt="coo", transpose=True),
        hg.adj(etype='ef', scipy_fmt="coo", transpose=True),
    ]

    print(hg)

    # node2feature的feature_index 在此留了padding，即编号从1开始
    p_features = p_vs_t.toarray()
    p_features = multihot2index(p_features)
    bias = p_features.max() + 1
    if onehot_for_nofeature:
        # e / a / f feature 都使用onehot
        e_features = np.arange(bias, hg.nodes('hyperedge').shape[0]+bias)[:, np.newaxis]
        bias += hg.nodes('hyperedge').shape[0]
        a_features = np.arange(bias, hg.nodes('author').shape[0]+bias)[:, np.newaxis]
        bias += hg.nodes('author').shape[0]
        f_features = np.arange(bias, hg.nodes('field').shape[0]+bias)[:, np.newaxis]
        bias += hg.nodes('field').shape[0]
    else:
        # 补一个padding
        e_features = np.zeros(hg.nodes('hyperedge').shape[0])[:, np.newaxis]
        a_features = np.zeros(hg.nodes('author').shape[0])[:, np.newaxis]
        f_features = np.zeros(hg.nodes('field').shape[0])[:, np.newaxis]

    features_dict = {
        "e_features": e_features,
        "p_features": p_features,
        "a_features": a_features,
        "f_features": f_features,
    }

    E4N_adjs, features, feature_types = [], [features_dict['e_features']], ["e_features",]
    for node_type in ['p', 'a', 'f']:   # 保持顺序
        if node_type not in node_types:
            continue
        E4N_adjs.append(hg.adj(etype='e{}'.format(node_type), scipy_fmt="coo", transpose=True))
        feature_types.append("{}_features".format(node_type))
        features.append(features_dict[feature_types[-1]])

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = labels.astype(np.int64)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_labeled_nodes = hg.number_of_nodes('hyperedge')
    train_mask = get_binary_mask(num_labeled_nodes, train_idx)
    val_mask = get_binary_mask(num_labeled_nodes, val_idx)
    test_mask = get_binary_mask(num_labeled_nodes, test_idx)

    return E4N_adjs, features, labels, num_classes, feature_types, \
           train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_acm_raw_graph(onehot_for_nofeature=True, node_types="paf"):
    """
    This code & dataset is modified based on DGL library.
    Thanks to https://github.com/dmlc/dgl
    """
    url = 'dataset/ACM.mat'
    data_path = './Dataset/ACM/ACM.mat'
    # try:
    #     data = sio.loadmat(data_path)
    # except FileNotFoundError:
    download(_get_dgl_url(url), path=data_path, overwrite=False)
    data = sio.loadmat(data_path)

    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
    a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
    t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
    # c_selected = (p_vs_c[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected].T[l_selected].T
    p_vs_a = p_vs_a[p_selected].T[a_selected].T
    p_vs_t = p_vs_t[p_selected].T[t_selected].T
    p_vs_c = p_vs_c[p_selected]  # .T[c_selected].T  # 这里不改是为了后面label正确

    hg = dgl.heterograph({
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
    })
    E4N_adjs = [
        hg.adj(etype='pa', scipy_fmt="coo", transpose=True),
        hg.adj(etype='pf', scipy_fmt="coo", transpose=True),
    ]

    print(hg)

    # node2feature的feature_index 在此留了padding，即编号从1开始
    p_features = p_vs_t.toarray()
    p_features = multihot2index(p_features)
    bias = p_features.max() + 1
    if onehot_for_nofeature:
        # e / a / f feature 都使用onehot
        a_features = np.arange(bias, hg.nodes('author').shape[0]+bias)[:, np.newaxis]
        bias += hg.nodes('author').shape[0]
        f_features = np.arange(bias, hg.nodes('field').shape[0]+bias)[:, np.newaxis]
        bias += hg.nodes('field').shape[0]
    else:
        # 补一个padding
        a_features = np.zeros(hg.nodes('author').shape[0])[:, np.newaxis]
        f_features = np.zeros(hg.nodes('field').shape[0])[:, np.newaxis]

    features_dict = {
        "p_features": p_features,
        "a_features": a_features,
        "f_features": f_features,
    }

    E4N_adjs, features, feature_types = [], [features_dict['p_features']], ["p_features",]
    for node_type in ['a', 'f']:   # 保持顺序
        if node_type not in node_types:
            continue
        E4N_adjs.append(hg.adj(etype='p{}'.format(node_type), scipy_fmt="coo", transpose=True))
        feature_types.append("{}_features".format(node_type))
        features.append(features_dict[feature_types[-1]])

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = labels.astype(np.int64)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_labeled_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_labeled_nodes, train_idx)
    val_mask = get_binary_mask(num_labeled_nodes, val_idx)
    test_mask = get_binary_mask(num_labeled_nodes, test_idx)

    return E4N_adjs, features, labels, num_classes, feature_types, \
           train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_acm_raw(onehot_for_nofeature=True, node_types="paf", hypergraph=True):
    if hypergraph:
        return load_acm_raw_hypergraph(onehot_for_nofeature, node_types)
    else:
        return load_acm_raw_graph(onehot_for_nofeature, node_types)



if __name__ == '__main__':
    import os

    # os.getcwd()
    os.chdir("../")

    E4N_adjs, features, labels, num_classes, feature_types, \
    train_idx, val_idx, test_idx, \
    train_mask, val_mask, test_mask = load_acm_raw()