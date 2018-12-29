import numpy as np


class Map:

    def AP(self, ranked_list, image_label_path):
        """
        Compute the average precision (AP) of a list of ranked items
        :param ranked_list:
        :return: AP
        """
        image_label = np.loadtxt(image_label_path, dtype=int)
        hits = 0
        sum_precs = 0
        for i, item in enumerate(ranked_list):
            if image_label[item] == 1:
                hits += 1
                sum_precs += hits / (i + 1.0)
        return sum_precs / len(ranked_list)

