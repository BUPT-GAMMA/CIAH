# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

def multihot2index(feature_mat):
    # np.insert 是为了给feature补个padding，即特征索引从 1 开始
    padded_feat = np.insert(feature_mat, 0, 0, axis=1)
    # multihot编码转为索引（长度不同）
    ans = MultiLabelBinarizer().fit([range(padded_feat.shape[1])]).inverse_transform(padded_feat)
    # 利用pandas对齐
    ans = pd.DataFrame(ans).fillna(0).values.astype(np.int64)
    return ans