# -*- coding: utf-8 -*-

from numpy import random
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
import data_set_generator


class Classification:
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter
        self.errors_ = []
        self.w_ = []

    @staticmethod
    def generate_random_weight(num_elem):
        return map(lambda r: r - 0.5, random.rand(num_elem))

    @staticmethod
    def insert_dummy_elem(x):
        return map(lambda v: [1] + v, x)

    @staticmethod
    def activation_func(z):
        if z > 0:
            return 1
        else:
            return -1

    def calculate_labels(self, data):
        data_with_dummy = self.insert_dummy_elem(data)
        labels = []
        for j in range(len(data_with_dummy)):
            label = self.calculate_label(data_with_dummy[j])
            labels.append(label)
        return labels

    def calculate_label(self, data_i):
        z = 0
        for j in range(len(data_i)):
            z += data_i[j] * self.w_[j]
        return self.activation_func(z)

    def print_result(self, data, labels):
        u"""2次元ベクトルを想定"""
        distinct_labels = list(set(labels))
        data_and_labels = zip(data, labels)
        colors = ["red", "blue", "green", "orange", "navy", "purple", "yellow", "violet", "tomato", "cyan", "pink"]
        plt.subplot(1, 2, 1)
        plt.title("result of classification")
        for label, color in zip(distinct_labels, colors):
            x = [d[0][0] for d in data_and_labels if d[1] == label]
            y = [d[0][1] for d in data_and_labels if d[1] == label]
            plt.scatter(x, y, color=color, label=label)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(list(range(1, len(self.errors_) + 1)), self.errors_, marker='o')
        plt.xlabel('epochs')
        plt.ylabel("number of mis-classifications")
        plt.xlim(0)
        plt.ylim(0)
        plt.show()


class ClassificationTest:
    def __init__(self, classification, training_data_rate=0.5):
        self.classification = classification
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

        self.classification.fit(training_data, training_labels)
        result_labels = self.classification.calculate_labels(test_data)

        self.classification.print_result(test_data, result_labels)
