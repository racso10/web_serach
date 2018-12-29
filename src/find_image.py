from findBest import *


def find_image(item, k):
    """
    搜索图片
    :param item: 单词
    :param k: 搜索结果的图片张数
    :return:
    """
    data_path = "../base/text8"
    w = WordSim(data_path)  # 词相似性计算模型类
    w.load_model()

    f = FindBest(w)  # 搜索图像类
    f.read('../data/denoiseTags_low10_result.txt')
    mr_rank = MR_rank('closed form')
    score_index, score = f.word2image(item, k)
    print("have found")
    f.show_image(score_index, '../data/Train_imageOutPutFileList.txt')
    m = Map()
    image_label_path = 'D:\\file\\作业\\web搜索\\data\\NUS-WIDE-Lite\\NUS-WIDE-Lite\\NUS-WIDE-Lite_groundtruth\\Lite_Labels_' + str(
        item) + '_Train.txt'
    print(m.AP(score_index, image_label_path))
    new_score_index = mr_rank.work(score_index, score, k)
    print("MP ranking done!")
    f.show_image(new_score_index, '../data/Train_imageOutPutFileList.txt')
    print(m.AP(new_score_index, image_label_path))


k = 50
item = 'plane'
find_image(item, k)
