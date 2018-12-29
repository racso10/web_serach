import logging
from gensim.models import word2vec


class WordSim:

    def __init__(self, path):
        """

        :param path: 训练数据路径
        """
        self.path = path
        self.model_path = path + '.model'  # 模型保存路径
        self.model = None

    def train(self):
        """
        训练模型
        :return: 模型保存路径
        """
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.Text8Corpus("../base/text8")  # 加载语料
        model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5

        # # 计算两个词的相似度/相关程度
        # y1 = model.similarity("woman", "man")
        # print(u"woman和man的相似度为：", y1)
        # print("--------\n")
        #
        # # 计算某个词的相关词列表
        # y2 = model.most_similar("good", topn=20)  # 20个最相关的
        # print(u"和good最相关的词有：\n")
        # for item in y2:
        #     print(item[0], item[1])
        # print("--------\n")
        #
        # # 寻找对应关系
        # print
        # ' "boy" is to "father" as "girl" is to ...? \n'
        # y3 = model.most_similar(['girl', 'father'], ['boy'], topn=3)
        # for item in y3:
        #     print
        #     item[0], item[1]
        # print
        # "--------\n"
        #
        # more_examples = ["he his she", "big bigger bad", "going went being"]
        # for example in more_examples:
        #     a, b, x = example.split()
        #     predicted = model.most_similar([x, b], [a])[0][0]
        #     print
        #     "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
        # print
        # "--------\n"
        #
        # # 寻找不合群的词
        # y4 = model.doesnt_match("breakfast cereal dinner lunch".split())
        # print
        # u"不合群的词：", y4
        # print
        # "--------\n"

        # 保存模型，以便重用
        model.save(self.model_path)
        return self.model_path

    def load_model(self):
        """
        加载模型
        :return:
        """
        self.model = word2vec.Word2Vec.load(self.model_path)

    def test(self, word1, word2):
        """
        计算两个词的相似度/相关程度
        :param word1:
        :param word2:
        :return:
        """
        try:
            return self.model.similarity(word1, word2)
        except KeyError:
            return 0


if __name__ == "__main__":
    data_path = "../base/text8"
    w = WordSim(data_path)
    w.load_model()

    print(w.test("car", "cars"))
    print(w.test("good", "bad"))
    print(w.test("computer", "computer"))
    print(w.test("photo", "image"))
