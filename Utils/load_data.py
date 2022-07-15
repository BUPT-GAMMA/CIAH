# -*- encoding: utf-8 -*-
from Utils.acm import load_acm_raw
from Utils.imdb import load_imdb
# from Utils.mt import load_mt
# from Utils.mt_ol import load_mt_ol
from Utils.multihot2index import multihot2index

import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict

def load_data(dataset, onehot_for_nofeature=True, hypergraph=True):
    if dataset.lower() == 'acm':
        E4N_adjs, features, labels, num_classes, feature_types, \
        train_idx, val_idx, test_idx, \
        train_mask, val_mask, test_mask = load_acm_raw(onehot_for_nofeature=onehot_for_nofeature,
                                                       node_types="paf",
                                                       hypergraph=hypergraph)

    elif dataset.lower() == 'imdb':
        E4N_adjs, features, labels, num_classes, feature_types, \
        train_idx, val_idx, test_idx, \
        train_mask, val_mask, test_mask = load_imdb(onehot_for_nofeature=onehot_for_nofeature,
                                                    node_types="mda",
                                                    hypergraph=hypergraph)

    else:
        raise KeyError("unknown dataset: {}".format(dataset))

    # 构建 E4N 和 N4E   4 means "from"   (E4N: 行代表E，列代表N)   行号和数字代表的序号，均从 1 开始，0为预留的pad
    E4N, N4E = [], []
    for e2n in E4N_adjs:
        # 给id补padding
        # 第一行也是padding
        eid = e2n.row + 1
        nid = e2n.col + 1

        ans_e4n = [list() for _ in range(e2n.shape[0] + 1)]
        ans_n4e = [list() for _ in range(e2n.shape[1] + 1)]
        for e, n in zip(eid, nid):
            ans_e4n[e].append(n)
            ans_n4e[n].append(e)

        ans_e4n = pd.DataFrame(ans_e4n).fillna(0).values.astype(np.int64)
        ans_n4e = pd.DataFrame(ans_n4e).fillna(0).values.astype(np.int64)
        E4N.append(ans_e4n)
        N4E.append(ans_n4e)

    # 给feature补padding
    # feature矩阵中的值，是从1开始编号的，即feature_index以留了padding
    # 这里补一行 0 ，即前面将边序号设置为从1开始后，这里的行号也应从1开始
    num_features = 0
    for i, f in enumerate(features):
        features[i] = np.concatenate([np.zeros((1, f.shape[1])), f], axis=0).astype(np.int64)
        num_features = max(num_features, f.max() + 1)

    # 给index和mask补padding
    train_idx, val_idx, test_idx = train_idx + 1, val_idx + 1, test_idx + 1
    train_mask, val_mask, test_mask = [np.insert(m, 0, False, axis=0) for m in [train_mask, val_mask, test_mask]]

    # 给label补padding
    labels = np.insert(labels, 0, 0, axis=0)

    return (E4N, N4E), features, feature_types, num_features, labels, num_classes, \
        train_idx, val_idx, test_idx, \
        train_mask, val_mask, test_mask



if __name__ == '__main__':
    import os

    # os.getcwd()
    os.chdir("../../scene_mining_new/")
    # (E4N, N4E), features, feature_types, num_features, labels, num_classes, \
        # train_idx, val_idx, test_idx, \
        # train_mask, val_mask, test_mask = load_data("acm")
    otpt = load_data("mt_ol")
    (E4N, N4E), features, feature_types, num_features, labels, num_classes, \
        train_idx, val_idx, test_idx, \
        train_mask, val_mask, test_mask =  otpt
    from Utils import metrics

    nmi = metrics.cluster_nmi(y_true=labels, y_pred=np.random.randint(num_classes, size=labels.shape))