from numpy import random

eta = 0.5

def generate_random_weight(num_elem):
    return random.rand(num_elem)

def insert_dummy_elem(x):
    map(lambda v: v.insert(0, 1), x)

def phi(z):
    if (z > 0):
        return 1
    else:
        return -1

if __name__ == '__main__':
    xs = [[1,2,3,4,5],
         [-5,-4,-3,-2,-1],
         [-6,-4,-2,-1,-1],
         [2,2,4,5,5]]
    labels = [1, -1, -1, 1]

    insert_dummy_elem(xs)
    w = generate_random_weight(len(xs[0]))

    for time in range(1):
        for j in range(len(xs)):
            z = 0
            for i in range(len(xs[0])):
                z += xs[j][i] * w[i]
            dw = eta * (labels[j] - phi(z)) * xs[j][i]
            w[i] += dw

    for j in range(len(xs)):
        z = 0
        for i in range(len(xs[0])):
            z += xs[j][i] * w[i]
        print phi(z)
