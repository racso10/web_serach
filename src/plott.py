"""
@project: search
@author: Racso
@file: plott.py
@ide: PyCharm
@date: 2018-12-20 10:56:36
"""

import matplotlib.pyplot as plt
import numpy as np

Concepts81 = np.loadtxt('D:\\file\\作业\\web搜索\\data\\ConceptsList\\Concepts81.txt', dtype=str)
score_sum = np.load('score_sum_top2.npy')
new_score_sum = np.load('new_score_sum_top2.npy')

total_width, n = 0.8, 2
width = total_width / n
new_range = []
plt.bar(range(81), score_sum, width=width, label='rank0')
for i in range(81):
    new_range.append(i + width)
plt.bar(new_range, new_score_sum, width=width, tick_label=Concepts81.tolist(), label='rerank')
ax = plt.gca()
for label in ax.xaxis.get_ticklabels():
    label.set_rotation(90)
# plt.(Concepts81.tolist(), rotation=45)
plt.legend()
plt.show()

# plt.bar(range(81), (new_score_sum - score_sum) / score_sum, tick_label=Concepts81.tolist(), label='rerank')
# ax = plt.gca()
# for label in ax.xaxis.get_ticklabels():
#     label.set_rotation(90)
# plt.legend()
# plt.show()
print(sum(score_sum) / 81)
print(sum(new_score_sum) / 81)
