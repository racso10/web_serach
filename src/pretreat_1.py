from spellchecker import SpellChecker

spell = SpellChecker()  # 拼写检查器
count = 0  # 统计错误单词个数
num = 0  # 所有单词数


class Pret:

    def __init__(self, nuswide):
        """
        数据预处理：去除非英文tag，去除过长、过短单词
        :param nuswide: tags路径
        """
        self.nuswide = nuswide
        self.info = []
        self.image_info = []

    def read(self):
        """
        读文件
        :return: 按行读取出的字符串
        """
        f = open(nuswide, 'r', encoding='UTF-8')
        self.info = f.readlines()

    def work(self):
        """

        :return: 处理后的list
        """
        self.read()
        for item in self.info:
            item_list = item[:-1].split(' ')
            image_info_tmp = []
            for word in item_list[6:]:
                if judge_pure_english(word) and 2 < len(word) < 10:
                    image_info_tmp.append(word)
            #   self.image_info.append([item_list[0]] + image_info_tmp)
            image_info_correct = is_correct(image_info_tmp)
            self.image_info.append([item_list[0]] + image_info_correct)
            if len(self.image_info) % 10000 == 0:
                print(len(self.image_info))
                print('wrong ---', count)
                print('all ---', num)
        print('wrong ---', count)
        print('all ---', num)
        return self.image_info

    def work_correct(self):
        """

        :return: 处理后的list
        """
        self.read()
        for item in self.info:
            item_list = item[:-1].split(' ')
            image_info_tmp = []
            for word in item_list[6:]:
                if judge_pure_english(word) and 2 < len(word) < 10:
                    image_info_tmp.append(word)
            #   self.image_info.append([item_list[0]] + image_info_tmp)
            image_info_correct = get_correct(image_info_tmp)
            self.image_info.append([item_list[0]] + image_info_correct)
            if len(self.image_info) % 100 == 0:
                print(len(self.image_info))
                print('wrong ---', count)
                print('all ---', num)
        print('wrong ---', count)
        print('all ---', num)
        return self.image_info


def judge_pure_english(keyword):
    """
    判断单词是否为纯英文
    :param keyword:待识别的单词
    :return: 是返回True 不是返回False
    """
    return all(is_alphabet(c) for c in keyword)


def is_alphabet(uchar):
    """
    判断一个unicode是否是英文字母
    :param uchar:待识别的字符
    :return: 是返回True 不是返回False
    """
    if u'\u0041' <= uchar <= u'\u005a' or u'\u0061' <= uchar <= u'\u007a':
        return True
    else:
        return False


def is_correct(list):
    """
   判断一个list中单词是否正确
   :param list:待识别的单词列表
   :return: 正确的单词列表
   """
    temp = []
    rightlist = spell.known(list)  # {'morning'}
    global count
    global num
    count = count + len(list) - len(rightlist)
    num = num + len(list)
    for item in rightlist:
        temp.append(item)
    return temp


def get_correct(list):
    """
   将一个list中错误单词改正后返回
   :param list:待识别的单词列表
   :return: 改正之后的单词列表
   """
    temp = []
    rightlist = spell.known(list)  # {'morning'}
    for item in rightlist:
        temp.append(item)

    wronglist = spell.unknown(list)
    global count
    count += len(wronglist)
    # print('*****', len(wronglist))
    for item in wronglist:
        item_tmp = spell.correction(item)
        temp.append(item_tmp)
    return temp


def write(file_name, data_list):
    """

    :param file_name: 保存的文件名
    :param data_list: 带保存list
    :return:
    """
    fp = open(file_name, 'w')
    for item in data_list:
        for j in item:
            fp.write(j + ' ')
        fp.write('\n')
    fp.close()


if __name__ == "__main__":
    nuswide = "C:\\Users\\WQ\\Desktop\\搜索大作业\\data\\NUS_WID_Tags\\All_Tags.txt"  # the location of your nus-wide-urls.
    # nuswide = "C:\\Users\\WQ\\Desktop\\搜索大作业\\data\\denoiseTags_low10.txt"  # the location of your nus-wide-urls.txt
    s = Pret(nuswide)
    write('denoiseTags_low10_result.txt', s.work())
    # write('file_name_correct.txt', s.work_correct())
