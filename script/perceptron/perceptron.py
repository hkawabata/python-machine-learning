# -*- coding: utf-8 -*-

from numpy import random
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../data_set_generator')
import data_set_generator


eta = 0.01

errors = []

def generate_random_weight(num_elem):
    return map(lambda r: r - 0.5, random.rand(num_elem))


def insert_dummy_elem(x):
    return map(lambda v: [1] + v, x)


def activation_func(z):
    if z > 0:
        return 1
    else:
        return -1


def execute(training_data, training_labels):
    training_data_with_dummy = insert_dummy_elem(training_data)
    w = generate_random_weight(len(training_data_with_dummy[0]))

    for time in range(100):
        cnt_err = 0
        for xi, labeli in zip(training_data_with_dummy, training_labels):
            update = eta * (labeli - calculate_label(xi, w))

            for j in range(len(w)):
                w[j] += update * xi[j]
            if update != 0:
                cnt_err += 1
        if cnt_err == 0:
            print "{0} errors".format(cnt_err)
            print "converged after {0} trial".format(time)
            break
        else:
            print "{0} errors".format(cnt_err)
    return w


def calculate_labels(data, w):
    data_with_dummy = insert_dummy_elem(data)
    labels = []
    for j in range(len(data_with_dummy)):
        label = calculate_label(data_with_dummy[j], w)
        labels.append(label)
    return labels


def calculate_label(data_i, w):
    z = 0
    for j in range(len(data_i)):
        z += data_i[j] * w[j]
    return activation_func(z)


def print_result(data, labels):
    u"""2次元ベクトルを想定"""
    distinct_labels = list(set(labels))
    data_and_labels = zip(data, labels)
    colors = ["red", "blue", "green", "orange", "navy", "purple", "yellow", "violet", "tomato", "cyan", "pink"]
    for label, color in zip(distinct_labels, colors):
        x = [d[0][0] for d in data_and_labels if d[1] == label]
        y = [d[0][1] for d in data_and_labels if d[1] == label]
        plt.scatter(x, y, color=color, label=label)
    plt.legend(loc='upper left')
    plt.show()


def test():
    training_data = [[1,2],[2,3],[1,4],[-1,0],[-5,-3],[-2,-1],[4,5],[-6,-5],
                    [1,0],[2,1],[1,-3],[-1,-3],[-5,-8],[-2,-4],[4,3],[-6,-7]]
    training_labels = [1,1,1,1,1,1,1,1,
              -1,-1,-1,-1,-1,-1,-1,-1]
    test_data = [[1,4],[1,-2],[3,4],[4,2],[-1,8],[-1,-4],[-5,-2],[-5,-9],[-2,0],[-2,-3]]
    test_labels = [1,-1,1,-1,1,-1,1,-1,1,-1]

    w = execute(training_data, training_labels)
    labels = calculate_labels(test_data, w)
    print_result(test_data, labels)


def test2():
    data_set = data_set_generator.generate_two_circle_data_set(500, x1=1, y1=2, r1=1, x2=-1, y2=3, r2=1)
    data = [x[0] for x in data_set]
    labels = [x[1] for x in data_set]
    do_test(data, labels)


def test3():
    data_set = data_set_generator.generate_square_data_set(1000, 0, 0, 5, 5, 2)
    data = [x[0] for x in data_set]
    labels = [x[1] for x in data_set]
    do_test(data, labels)


def do_test(data, labels):
    num_of_traning_data = len(data)*1/3
    training_data = data[:num_of_traning_data]
    training_labels = labels[:num_of_traning_data]
    test_data = data[num_of_traning_data:]
    test_labels = labels[num_of_traning_data:]

    w = execute(training_data, training_labels)
    result_labels = calculate_labels(test_data, w)

    print_result(test_data, result_labels)


if __name__ == '__main__':
    test()
    test2()
    test3()
