# -*- coding: utf-8 -*-


class Classification:
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter
        self.errors_ = []
        self.w_ = []
