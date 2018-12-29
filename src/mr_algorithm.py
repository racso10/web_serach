import numpy as np


class MR:
    """
    流型排序
    """

    def __init__(self, k, S, f, alpha=0.99, max_step=10000000):
        """

        :param k: 重排序样本数
        :param S: 标准化矩阵 S=D^(1/2)WD（1/2）
        :param f: 初始排序
        :param alpha: 初始排序比重
        :param max_step: 最大迭代步数
        """
        self.y = np.zeros(k, dtype=float)
        self.S = S
        self.f = f
        self.alpha = alpha
        self.maxStep = max_step

    def init_y(self):
        """
        初始化 y
        :return:
        """
        self.y = self.f

    def work(self):
        """
        多次迭代，直到达到最大迭代步数或稳定，终止
        :return:
        """
        self.init_y()
        for step in range(self.maxStep):
            tmp_f = self.alpha * np.dot(self.S, self.f) + (1 - self.alpha) * self.y
            if np.sqrt(np.sum(tmp_f - self.f) ** 2) < 1e-4:
                print('break ', step)
                break
            self.f = tmp_f
            # self.f = self.alpha * np.dot(self.S, self.f) + (1 - self.alpha) * self.y
        return self.f


if __name__ == '__main__':
    k = 10
    S = np.random.rand(10, 10)
    f = np.sort(np.random.rand(10))
    alpha = 0.9
    print(np.dot(S, f))
    mr = MR(k, S, f)
    print(mr.work())
