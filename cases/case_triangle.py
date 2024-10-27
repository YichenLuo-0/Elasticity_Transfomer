import numpy as np
from overrides import overrides

from cases.elastic_body import ElasticBody


class Triangle(ElasticBody):
    def __init__(self, e, nu, l, q0):
        super().__init__(e, nu, 0, l, 0, l)
        self.l = l
        self.q0 = q0

    @overrides
    def geo_filter(self, x, y):
        return y <= x

    @overrides
    def bc_filter(self, x, y):
        # Face AB，受到向下的分布载荷
        if y == 0:
            qx = -x / self.l * self.q0
            bc = np.array([0, -1, 0, qx, 0, 0, 2])

        # Face BC，固定在墙上
        elif x == self.l:
            bc = np.array([0, 0, 0, 0, 0, 0, 1])

        # Face AC，不受力
        elif y == x:
            l_ = -1 / np.sqrt(2)
            m_ = 1 / np.sqrt(2)
            bc = np.array([l_, m_, 0, 0, 0, 0, 2])

        # 其他内部点无边界条件
        else:
            bc = np.array([0, 0, 0, 0, 0, 0, 0])
        return bc

    @overrides
    def ground_truth(self, x, y):
        # calculate the ground truth
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        sigma_x = -(self.q0 / self.l) * (x - (2 * y))
        sigma_y = (self.q0 / self.l) * x
        tau_xy = (self.q0 / self.l) * y

        print(sigma_x.shape)
        return u, v, sigma_x, sigma_y, tau_xy

    def set_load(self, q0):
        self.q0 = q0
        return self
