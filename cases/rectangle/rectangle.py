import numpy as np
from overrides import overrides

from cases.elastic_body import ElasticBody


class Rectangle(ElasticBody):
    def __init__(self, e, nu, l, q0):
        super().__init__(e, nu, 0, l, 0, l)
        self.l = l
        self.q0 = q0

    @overrides
    def geo_filter(self, x, y):
        return True

    @overrides
    def bc_filter(self, x, y):
        if x == 0:
            bc = np.array([0, -1, 0, qx, 0, 0, 1])

        elif x == 1:
            bc = np.array([0, 0, 0, 0, 0, 0, 1])

        elif y == 0:
            bc = np.array([0, 0, 0, 0, 0, 0, 2])

        elif y == 1:

            bc = np.array([l_, m_, 0, 0, 0, 0, 1])

        # 其他内部点无边界条件
        else:
            bc = np.array([0, 0, 0, 0, 0, 0, 0])
        return bc
