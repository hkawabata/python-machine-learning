# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from classifier import Classifier, ClassifierTest
import numpy as np


class AdalineGD(Classifier):
    """ADALINE 勾配降下法による分類器"""

    def __init__(self, eta, n_iter):
        Classifier.__init__(self, eta, n_iter)

    @staticmethod
    def standardize(x):
        """ベクトルを標準化（not 規格化）する"""
        x_array = np.array(x)
        res = np.copy(x_array)
        res[:, 0] = (x_array[:, 0] - x_array[:, 0].mean()) / x_array[:, 0].std()
        res[:, 1] = (x_array[:, 1] - x_array[:, 1].mean()) / x_array[:, 1].std()
        return res

    def fit(self, training_data, training_labels):
        training_data_with_dummy = self.insert_dummy_elem(training_data)
        """
        # 標準化後の座標空間で plot しないと結果がおかしいため、今はコメントアウト
        std = self.standardize([[float(y) for y in x] for x in training_data])
        training_data_with_dummy = self.insert_dummy_elem(std)
        """
        self.errors_ = []
        self.w_ = self.generate_random_weight(len(training_data_with_dummy[0]))
        for time in range(self.n_iter):
            label_diff = []
            for xi, correct_label in zip(training_data_with_dummy, training_labels):
                label_diff.append(correct_label - self.predict_label(xi))
            cnt_err = len([diff for diff in label_diff if diff != 0])
            self.errors_.append(cnt_err)
            if cnt_err == 0:
                break
            for j in range(len(self.w_)):
                dw = self.eta * training_data_with_dummy[:, j].dot(label_diff)
                self.w_[j] += dw


class AdalineGDLargeData(AdalineGD):
    """ADALINE 確率勾配降下法による分類器. データが大規模な場合に使用"""

    def __init__(self, eta, n_iter):
        AdalineGD.__init__(self, eta, n_iter)

    def fit(self, training_data, training_labels):
        """確率的勾配降下法で学習を行う"""
        data = self.insert_dummy_elem(training_data)
        labels = training_labels
        self.errors_ = []
        self.w_ = self.generate_random_weight(len(data[0]))
        for time in range(self.n_iter):
            cnt_err = 0
            for xi, correct_label in zip(data, labels):
                label_diff = correct_label - self.predict_label(xi)
                dw = self.eta * label_diff * xi
                self.w_ += dw
                if label_diff != 0:
                    cnt_err += 1
            self.errors_.append(cnt_err)
            if cnt_err == 0:
                break
            data, labels = self.shuffle_data_and_labels(data, labels)

    @staticmethod
    def shuffle_data_and_labels(data, labels):
        """data, label をシャッフルする"""
        shuffled_data, shuffled_labels = zip(*np.random.permutation(zip(data, labels)))
        return np.array(shuffled_data), np.array(shuffled_labels)


if __name__ == '__main__':
    adaline_gd = AdalineGD(eta=0.001, n_iter=200)
    adaline_gd_test = ClassifierTest(adaline_gd, training_data_rate=1.0 / 3)
    adaline_gd_test.test()
    adaline_gd_test.test2()
    adaline_gd_test.test3()

    adaline_gd_large = AdalineGDLargeData(eta=0.001, n_iter=200)
    adaline_gd_large_test = ClassifierTest(adaline_gd_large, training_data_rate=1.0 / 3)
    adaline_gd_large_test.test()
    adaline_gd_large_test.test2()
    adaline_gd_large_test.test3()
