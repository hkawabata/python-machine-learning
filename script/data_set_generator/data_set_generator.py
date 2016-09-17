# -*- coding: utf-8 -*-

import numpy.random
import random


def generate_two_circle_data_set(num, x1=1, y1=1, r1=1, x2=-1, y2=-1, r2=1):
    data_set = []
    for i in range(num / 2):
        x = numpy.random.rand() * r1 * 2 - r1
        y = numpy.random.rand() * r1 * 2 - r1
        while x * x + y * y > r1:
            x = numpy.random.rand() * r1 * 2 - r1
            y = numpy.random.rand() * r1 * 2 - r1
        data_set.append(([x + x1, y + y1], 1))
    for i in range(num - num / 2):
        x = numpy.random.rand() * r2 * 2 - r2
        y = numpy.random.rand() * r2 * 2 - r2
        while x * x + y * y > r2:
            x = numpy.random.rand() * r2 * 2 - r2
            y = numpy.random.rand() * r2 * 2 - r2
        data_set.append(([x + x2, y + y2], -1))
    random.shuffle(data_set)
    return data_set


def generate_square_data_set(num, x1, y1, x2, y2, a):
    x_width = x2 - x1
    y_width = y2 - y1
    x = [x1 + _x * x_width for _x in numpy.random.rand(num)]
    y = [y1 + _y * y_width for _y in numpy.random.rand(num)]
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    data_set = []
    for _x, _y in zip(x, y):
        label = 1 if a * (_x - x_center) + y_center < _y else -1
        data_set.append(([_x, _y], label))
    return data_set
