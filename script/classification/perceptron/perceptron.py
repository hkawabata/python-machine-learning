# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from classification import Classification, ClassificationTest


class Perceptron(Classification):
    """パーセプトロンの分類器

    パラメータ
    ----------
    eta : float
        学習率 (0, 1.0]
    n_iter : int
        トレーニングの最大反復回数

    属性
    ----------
    w_ : 適合後の重み
    errors_ : 各エポックでの誤分類の数
    """

    def __init__(self, eta, n_iter):
        Classification.__init__(self, eta, n_iter)

    def fit(self, training_data, training_labels):
        training_data_with_dummy = self.insert_dummy_elem(training_data)
        self.errors_ = []
        self.w_ = self.generate_random_weight(len(training_data_with_dummy[0]))

        for time in range(self.n_iter):
            cnt_err = 0
            for xi, labeli in zip(training_data_with_dummy, training_labels):
                update = self.eta * (labeli - self.calculate_label(xi))

                for j in range(len(self.w_)):
                    self.w_[j] += update * xi[j]
                if update != 0:
                    cnt_err += 1
            self.errors_.append(cnt_err)
            if cnt_err == 0:
                print "{0} errors".format(cnt_err)
                print "converged after {0} trial".format(time)
                break
            else:
                print "{0} errors".format(cnt_err)


if __name__ == '__main__':
    ppn = Perceptron(eta=0.01, n_iter=100)
    ppn_test = ClassificationTest(ppn, training_data_rate=1.0/3)
    ppn_test.test()
    ppn_test.test2()
    ppn_test.test3()
