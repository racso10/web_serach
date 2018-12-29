import matplotlib.pyplot as plt
from PIL import Image

from wordSimilar import *
from Manifold_ranking import *


class FindBest:
    """
    通过一个单词找到最符合的Top_k图片
    """

    def __init__(self, w):
        """

        :param w: 训练好的word2vec模型
        """
        self.image_tag_path = ''
        self.image_list_path = ''
        self.info = []  # 每张图片的Tags
        self.w = w

    def read(self, image_tag_path):
        """
        读文件
        :return: 按行读取出的字符串
        """
        self.image_tag_path = image_tag_path
        f = open(self.image_tag_path, 'r', encoding='UTF-8')
        self.info = f.readlines()

    def word_cmp_tags(self, word, Tags):
        """
        单词和图片Tags的计算相似度
        :param word: 待比较单词
        :param Tags: 图像的Tags
        :return: 相似度
        """
        score = []
        for item in Tags:
            tmp = self.w.test(word, item)
            score.append(tmp * tmp)
            # score.append(np.exp(tmp))
            # score.append(tmp)

        score.sort(reverse=True)
        return sum(score[:2]) / len(score[:2])  # 取相似度最高的k个词做平均，避免因为标签个数不同带来的影响

    def word2image(self, word, k):
        """
        通过单词找到最符合的Top_k图片
        :param word: 单词
        :param k: k
        :return: Top_k图片的标号
        """
        score = []
        for item in self.info:
            Tags = item[:-1].split(' ')[1:]  # 去掉图片ID
            score.append(self.word_cmp_tags(word, Tags))
        score = np.array(score)
        score_index = score.argsort()[::-1]
        score = np.sort(score)[::-1]
        return score_index[:k], score[:k]

    def show_image(self, score_index, image_list_path):
        """
        显示Top_k图片
        :param score_index: Top_k图片的标号
        :param image_list_path: 图片路径
        :return:
        """
        self.image_list_path = image_list_path

        imgae_list = np.loadtxt(self.image_list_path, dtype=str)
        image = imgae_list.take(score_index, axis=0)
        i = 1
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.07, hspace=0.07)
        for item in image:
            image_path = 'D:\\file\\作业\\web搜索\\data\\Flickr\\Flickr\\Flickr\\' + item
            img = Image.open(image_path)
            plt.subplot(8, 7, i)
            i += 1
            plt.imshow(img)
            plt.axis('off')
        plt.show()


def cmp_form_iteration():
    # k = 50
    item = 'sky'
    mr_feature = 'CH'
    # mr_func = 'closed form'
    # mr_func = 'iteration'
    mr_sigma = 0.2
    mr_alpha = 0.99

    data_path = "../base/text8"
    w = WordSim(data_path)
    w.load_model()

    f = FindBest(w)
    f.read('../data/denoiseTags_low10_result.txt')

    score_index, score = f.word2image(item, 10000)

    time_form = []
    time_iteration = []
    acc = []
    m = Map()
    image_label_path = 'D:\\file\\作业\\web搜索\\data\\NUS-WIDE-Lite\\NUS-WIDE-Lite\\NUS-WIDE-Lite_groundtruth\\Lite_Labels_' + str(
        item) + '_Train.txt'
    for i in range(100, 3000, 100):
        mr_rank = MR_rank('closed form')
        new_score_index_form, time_cost = mr_rank.work(score_index[:i], score[:i], i, mr_feature, sigma=mr_sigma,
                                                       alpha=mr_alpha)
        score_form = m.AP(new_score_index_form, image_label_path)
        time_form.append(time_cost)

        mr_rank = MR_rank('iteration')
        new_score_index_iteration, time_cost = mr_rank.work(score_index[:i], score[:i], i, mr_feature, sigma=mr_sigma,
                                                            alpha=mr_alpha)

        score_iteration = m.AP(new_score_index_iteration, image_label_path)
        time_iteration.append(time_cost)

        acc.append(abs(score_form - score_iteration) / score_form)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(range(100, 3000, 100), time_form, 'r')
    ax1.plot(range(100, 3000, 100), time_iteration, 'b')
    ax1.legend(('closed form', 'iteration'), loc='best')
    ax1.set_ylabel('time')
    ax1.set_xlabel('number of image')
    ax1.set_title("cmp_form_iteration")

    ax2 = ax1.twinx()
    ax2.plot(range(100, 3000, 100), acc, 'g')
    ax2.set_ylabel('acc')

    plt.show()


if __name__ == "__main__":
    # cmp_form_iteration()

    k = 50
    item = 'apple'
    mr_feature = 'CH'
    mr_func = 'closed form'
    # mr_func = 'iteration'
    mr_sigma = 1.25
    mr_alpha = 0.99

    data_path = "../base/text8"
    w = WordSim(data_path)
    w.load_model()

    f = FindBest(w)
    f.read('../data/denoiseTags_low10_result.txt')
    mr_rank = MR_rank(mr_func)
    score_index, score = f.word2image(item, k)
    print("have found")
    m = Map()
    image_label_path = 'D:\\file\\作业\\web搜索\\data\\NUS-WIDE-Lite\\NUS-WIDE-Lite\\NUS-WIDE-Lite_' \
                       'groundtruth\\Lite_Labels_' + str(item) + '_Train.txt'
    try:
        rank0_score = m.AP(score_index, image_label_path)
    except FileNotFoundError:
        print("Not in ground_truth")
        print("MP ranking done")
        new_score_index, _ = mr_rank.work(score_index, score, k, mr_feature, sigma=mr_sigma, alpha=mr_alpha)
        f.show_image(new_score_index, '../data/Train_imageOutPutFileList.txt')
    else:
        new_score_index, _ = mr_rank.work(score_index, score, k, mr_feature, sigma=mr_sigma, alpha=mr_alpha)
        print("MP ranking done!")
        rerank_score = m.AP(new_score_index, image_label_path)
        if rerank_score > rank0_score:
            f.show_image(new_score_index, '../data/Train_imageOutPutFileList.txt')
            # print(rerank_score)
        else:
            f.show_image(score_index, '../data/Train_imageOutPutFileList.txt')
            # print(rank0_score)
