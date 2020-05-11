# -*- coding: UTF-8 -*-
#!/usr/bin/env python
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: lsh.py
# @Author: limingwei
# @Date: 2020-05-05
# @Mail: plorylmw AT gmail.com
# +++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np


class LSHAlgo(object):
    def __init__(self, feat_dim, code_dim):
        self.feat_dim = feat_dim
        self.code_dim = code_dim

        self.random_projection = np.random.randn(self.feat_dim, self.code_dim)

    def gen_codes(self, feats, mean_feats):
        return np.sign((feats - mean_feats).dot(self.random_projection))

    def run(self, feats):
        mean_feats = np.mean(feats, axis=0)
        codes = self.gen_codes(feats, mean_feats)
        return codes


if __name__ == '__main__':
    feats = np.random.randn(10000, 128)
    LSH = LSHAlgo(feat_dim=128, code_dim=32)
    codes = LSH.run(feats)
    print(codes)
    print(codes.shape) # (10000, 32)
