# -*- coding: utf-8 -*-

import numpy.random
import random

def generate_two_circle_data_set(num):
    data_set = []
    for i in range(num / 2):
        x = numpy.random.rand() * 2 - 1
        y = numpy.random.rand() * 2 - 1
        while x * x + y * y > 1:
            x = numpy.random.rand() * 2 - 1
            y = numpy.random.rand() * 2 - 1
        data_set.append(([x + 1, y + 1], 1))
    for i in range(num - num / 2):
        x = numpy.random.rand() * 2 - 1
        y = numpy.random.rand() * 2 - 1
        while x * x + y * y > 1:
            x = numpy.random.rand() * 2 - 1
            y = numpy.random.rand() * 2 - 1
        data_set.append(([x - 1, y - 1], -1))
    random.shuffle(data_set)
    return data_set
