from findBest import *
from Manifold_ranking import *
import matplotlib.pyplot as plt

data_path = "../base/text8"
w = WordSim(data_path)
w.load_model()

f = FindBest(w)
f.read('../data/denoiseTags_low10_result.txt')

Concepts81 = np.loadtxt('D:\\file\\作业\\web搜索\\data\\ConceptsList\\Concepts81.txt', dtype=str)
score_sum = []
new_score_sum = []
mr_rank = MR_rank('closed form')
k = 50
mr_feature = 'CH'

for item in Concepts81:
    score_index, score = f.word2image(item, k)
    m = Map()
    image_label_path = 'D:\\file\\作业\\web搜索\\data\\NUS-WIDE-Lite\\NUS-WIDE-Lite\\NUS-WIDE-Lite_groundtruth\\Lite_Labels_' + str(
        item) + '_Train.txt'
    score_sum.append(m.AP(score_index, image_label_path))

    new_score_index, _ = mr_rank.work(score_index, score, k, mr_feature)
    new_score_sum.append(m.AP(new_score_index, image_label_path))
    # if score_sum[-1] < new_score_sum[-1]:
    print(item, score_sum[-1], new_score_sum[-1])

score_sum = np.array(score_sum)
new_score_sum = np.array(new_score_sum)
np.save('score_sum_top2.npy', score_sum)
np.save('new_score_sum_top2.npy', new_score_sum)


#
# score_sum = np.random.rand(81)
# new_score_sum = np.random.rand(81)
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

plt.show()
print(sum(score_sum) / 81)
print(sum(new_score_sum) / 81)
