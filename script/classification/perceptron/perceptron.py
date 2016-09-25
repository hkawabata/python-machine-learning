# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import data_set_generator
from classification import Classification


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


class PerceptronTest:

    def __init__(self, perceptron, training_data_rate=0.5):
        self.ppn = perceptron
        self.td_rate = training_data_rate

    def test(self):
        data = [[1,2],[2,3],[-5,-8],[-2,-4],[1,4],[-1,0],[1,-3],[-1,-3],
                [1,0],[2,1],[4,5],[-6,-5],[4,3],[-6,-7],[-5,-3],[-2,-1],
                [1,4],[1,-2],[3,4],[4,2],[-1,8],[-1,-4],[-5,-2],[-5,-9],[-2,0],[-2,-3]]
        labels = [1,1,-1,-1,1,1,-1,-1,
                  -1,-1,1,1,-1,-1,1,1,
                  1,-1,1,-1,1,-1,1,-1,1,-1]
        self.do_test(data, labels)

    def test2(self):
        data_set = data_set_generator.generate_two_circle_data_set(500, x1=1, y1=2, r1=1, x2=-1, y2=3, r2=1)
        data = [x[0] for x in data_set]
        labels = [x[1] for x in data_set]
        self.do_test(data, labels)

    def test3(self):
        data_set = data_set_generator.generate_square_data_set(1000, x1=0, y1=0, x2=5, y2=5, a=2)
        data = [x[0] for x in data_set]
        labels = [x[1] for x in data_set]
        self.do_test(data, labels)

    def do_test(self, data, labels):
        num_of_traning_data = int(len(data) * self.td_rate)
        training_data = data[:num_of_traning_data]
        training_labels = labels[:num_of_traning_data]
        test_data = data[num_of_traning_data:]
        test_labels = labels[num_of_traning_data:]

        w = self.ppn.fit(training_data, training_labels)
        result_labels = self.ppn.calculate_labels(test_data)

        self.ppn.print_result(test_data, result_labels)


if __name__ == '__main__':
    ppn = Perceptron(eta=0.01, n_iter=100)
    ppn_test = PerceptronTest(ppn, training_data_rate=1.0/3)
    ppn_test.test()
    ppn_test.test2()
    ppn_test.test3()
