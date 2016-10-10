# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
import data_set_generator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class ScikitLearnClassifier:
    def __init__(self, classifier):
        self.classifier = classifier

    @classmethod
    def get_iris_dataset(cls, training_size, standardize=False):
        from sklearn import datasets
        from sklearn.cross_validation import train_test_split

        iris = datasets.load_iris()
        x = iris.data[:, [2, 3]]
        y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=training_size, random_state=0)
        if standardize:
            x_train_std, x_test_std = cls.standardize(x_train, x_test)
            return x_train_std, x_test_std, y_train, y_test
        else:
            return x_train, x_test, y_train, y_test

    @staticmethod
    def standardize(x_train, x_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)
        return x_train_std, x_test_std

    def print_result(self, x_test, y_test, resolution=0.02):
        u"""各データのラベルを判定し、結果をグラフ化して出力する"""
        unique_labels = np.unique(y_test)
        colors = ["red", "blue", "green", "orange", "navy", "purple", "yellow", "violet", "tomato", "cyan", "pink"]
        cmap = ListedColormap(colors[:len(unique_labels)])

        # 決定領域を計算
        x1_min, x1_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        x2_min, x2_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        y_grid_pred = self.classifier.predict(np.array([x1_grid.ravel(), x2_grid.ravel()]).T).reshape(x1_grid.shape)

        plt.title('result of classification ({0})'.format(self.classifier.__class__.__name__))
        plt.xlabel('x')
        plt.ylabel('y')
        # 決定領域を図示
        plt.contourf(x1_grid, x2_grid, y_grid_pred, alpha=0.4, cmap=cmap)
        # テストデータのラベルをプロット
        for label, color in zip(unique_labels, colors):
            _x = [xy[0][0] for xy in zip(x_test, y_test) if xy[1] == label]
            _y = [xy[0][1] for xy in zip(x_test, y_test) if xy[1] == label]
            plt.scatter(_x, _y, color=color, label=label, marker='o')
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.legend(loc='upper left')
        plt.show()

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)


class ScikitLearnClassifierDemo:

    @staticmethod
    def demo(classifier, training_size):
        x_train, x_test, y_train, y_test = ScikitLearnClassifier.get_iris_dataset(training_size)
        sc = ScikitLearnClassifier(classifier)
        sc.fit(x_train, y_train)
        sc.print_result(x_test, y_test)

    @classmethod
    def demo_perceptron(cls):
        from sklearn.linear_model import Perceptron
        ppn = Perceptron(n_iter=10000, eta0=0.01, random_state=0, shuffle=True)
        cls.demo(classifier=ppn, training_size=0.5)

    @classmethod
    def demo_logidtic_regression(cls):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1000, random_state=0)
        cls.demo(classifier=lr, training_size=0.5)

    @classmethod
    def demo_ridge(cls):
        from sklearn.linear_model import RidgeClassifier
        rdg = RidgeClassifier(random_state=0)
        cls.demo(classifier=rdg, training_size=0.5)

if __name__ == '__main__':
    ScikitLearnClassifierDemo.demo_perceptron()
    ScikitLearnClassifierDemo.demo_logidtic_regression()
    ScikitLearnClassifierDemo.demo_ridge()
