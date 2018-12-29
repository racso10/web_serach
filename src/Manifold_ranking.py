# ***********************
# 重排序中W与S的计算
# Kunwwk 2018.12.13
# ***********************

import math
from mean_average_precision import *
from mr_algorithm import *


class MR_rank:

    def __init__(self, func):
        self.form = True
        if func == 'closed form':
            self.form = True
        elif func == 'iteration':
            self.form = False

    def manifold_rank(self, data_rank, feature, sigma):
        data_set = np.loadtxt('../data/Normalized_' + feature + '_Lite_Train.dat')  # 数据的读入
        # sigma = 1.25
        # sigma = 0.1
        W = np.zeros((len(data_rank), len(data_rank)))

        D_Squareroot = np.zeros((len(data_rank), len(data_rank)))
        for i in range(len(data_rank)):
            Sequence_numi = data_rank[i]  # 这里Sequence_numi为第i张图的序号
            tempi = data_set[Sequence_numi]  # 这里tempj为第j张图片在数据集中的特征
            for j in range(len(data_rank)):
                Sequence_numj = data_rank[j]
                tempj = data_set[Sequence_numj]
                W[i][j] = math.exp(-np.sum((tempi - tempj) ** 2) / (2 * sigma ** 2))
                # W[i][j] = math.exp(-np.sum((tempi - tempj) ** 2) / (2 * sigma))
        for i in range(len(W)):
            D_Squareroot[i][i] = 1 / np.square(np.sum(W[i][:]))  # 其为实对角矩阵，故求逆为各个元素的倒数
        S = D_Squareroot * W * D_Squareroot
        return S

    def work(self, score_index, score, k, feature, sigma=1.25, alpha=0.99):

        S = self.manifold_rank(score_index, feature, sigma=sigma)

        import time
        start = time.time()
        if self.form:
            I = np.eye(k)
            new_scores = np.dot(np.linalg.inv(I - alpha * S), score)
        else:
            mr = MR(k, S, score)
            new_scores = mr.work()
        end = time.time()
        # print(end-start)

        new_scores_index = new_scores.argsort()

        new_scores_index = (k - new_scores_index - 1).tolist()
        new_score_index = []
        for i in range(k):
            p = new_scores_index.index(i)
            new_score_index.append(score_index[p])

        return new_score_index, end - start
