# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
import data_set_generator


class Classifier:
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter
        self.errors_ = []
        self.w_ = []

    @staticmethod
    def generate_random_weight(num_elem):
        return np.random.rand(num_elem) - 0.5

    @staticmethod
    def insert_dummy_elem(x):
        return np.array([[1.0] + list(xi) for xi in x])

    @staticmethod
    def activation_func(z):
        if z > 0:
            return 1
        else:
            return -1

    def predict_labels(self, data):
        data_with_dummy = self.insert_dummy_elem(data)
        labels = []
        for data in data_with_dummy:
            label = self.predict_label(data)
            labels.append(label)
        return np.array(labels)

    def predict_label(self, data_i):
        z = 0
        for j in range(len(data_i)):
            z += data_i[j] * self.w_[j]
        return self.activation_func(z)

    def print_result(self, data, resolution=0.02):
        u"""2次元ベクトルを想定"""
        # データのラベルを計算
        data_labels = self.predict_labels(data)
        # 等高線用のグリッドデータを作成し、各ブロックのラベルを計算
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
        grid = np.array(zip(x_grid.ravel(), y_grid.ravel()))
        grid_labels = self.predict_labels(grid).reshape(x_grid.shape)

        unique_labels = np.unique(data_labels)
        data_and_labels = zip(data, data_labels)
        colors = ["red", "blue", "green", "orange", "navy", "purple", "yellow", "violet", "tomato", "cyan", "pink"]
        cmap = ListedColormap(colors[:len(unique_labels)])

        # 分類結果
        plt.subplot(1, 2, 1)
        plt.title('result of classification')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left')
        plt.contourf(x_grid, y_grid, grid_labels, alpha=0.4, cmap=cmap)
        for label, color in zip(unique_labels, colors):
            x = [d[0][0] for d in data_and_labels if d[1] == label]
            y = [d[0][1] for d in data_and_labels if d[1] == label]
            plt.scatter(x, y, color=color, label=label)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # 誤分類の数の遷移
        plt.subplot(1, 2, 2)
        plt.xlabel('epochs')
        plt.ylabel('number of mis-classifications')
        plt.plot(list(range(1, len(self.errors_) + 1)), self.errors_, marker='o')
        plt.xlim(0)
        plt.ylim(0)
        plt.show()


class ClassifierTest:
    def __init__(self, classifier, training_data_rate=0.5):
        self.classifier = classifier
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
        data_set = data_set_generator.generate_square_data_set(500, x1=0, y1=0, x2=5, y2=5, a=2)
        data = [x[0] for x in data_set]
        labels = [x[1] for x in data_set]
        self.do_test(data, labels)

    def do_test(self, data, labels):
        num_of_traning_data = int(len(data) * self.td_rate)
        training_data = np.array(data[:num_of_traning_data])
        training_labels = np.array(labels[:num_of_traning_data])
        test_data = np.array(data[num_of_traning_data:])
        test_labels = np.array(labels[num_of_traning_data:])

        self.classifier.fit(training_data, training_labels)
        self.classifier.print_result(test_data)
