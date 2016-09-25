# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from classifier import Classifier, ClassifierTest
import numpy as np


class AdalineGD(Classifier):

    def __init__(self, eta, n_iter):
        Classifier.__init__(self, eta, n_iter)

    @staticmethod
    def standardize(x):
        x_array = np.array(x)
        res = np.copy(x_array)
        res[:,0] = (x_array[:,0] - x_array[:,0].mean()) / x_array[:,0].std()
        res[:,1] = (x_array[:,1] - x_array[:,1].mean()) / x_array[:,1].std()
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


if __name__ == '__main__':
    adaline_gd = AdalineGD(0.001, 1000)
    adaline_gd_test = ClassifierTest(adaline_gd, training_data_rate=1.0 / 3)
    adaline_gd_test.test()
    adaline_gd_test.test2()
    adaline_gd_test.test3()
