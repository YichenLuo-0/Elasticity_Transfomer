import numpy as np
from overrides import overrides

from cases.elastic_body import ElasticBody


class Triangle(ElasticBody):
    def __init__(self, e, nu, l, q0):
        super().__init__(e, nu)
        self.l = l
        self.q0 = q0

    @overrides
    def discretize(self, nx, ny):
        # generate points
        x, y = np.meshgrid(np.linspace(0, self.l, nx), np.linspace(0, self.l, ny))
        x = x.flatten()
        y = y.flatten()

        # mask the points outside the triangle
        mask = y <= x
        return x[mask], y[mask]

    @overrides
    def boundary_conditions(self, x, y):
        bc = np.zeros([x.shape[0], 7])

        # 生成边界条件
        # 边界条件编码格式：[l, m, tx, ty, u, v, mask]
        # l, m: 法向量
        # tx, ty: 对于力边界，tx, ty为力的分量
        # u, v: 对于位移边界，u, v为位移的分量
        for i in range(x.shape[0]):
            x_idx = x[i]
            y_idx = y[i]

            # Face AB，受到向下的分布载荷
            if y_idx == 0:
                qx = -x_idx / self.l * self.q0
                bc[i] = np.array([0, -1, 0, qx, 0, 0, 2])

            # Face BC，固定在墙上
            elif x_idx == self.l:
                bc[i] = np.array([0, 0, 0, 0, 0, 0, 1])

            # Face AC，不受力
            elif y_idx == x_idx:
                l_ = -1 / np.sqrt(2)
                m_ = 1 / np.sqrt(2)
                bc[i] = np.array([l_, m_, 0, 0, 0, 0, 2])

            # 其他内部点无边界条件
            else:
                bc[i] = np.array([0, 0, 0, 0, 0, 0, 0])
        return bc

    def set_load(self, q0):
        self.q0 = q0
        return self
