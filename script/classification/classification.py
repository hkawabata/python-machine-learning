# -*- coding: utf-8 -*-

from numpy import random
import matplotlib.pyplot as plt


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
