import numpy as np

from cases.elastic_body import ElasticBody


class Rectangle(ElasticBody):
    def __init__(self, e, nu, l, q0):
        super().__init__(e, nu)
        self.l = l
        self.q0 = q0

    def discretize(self, nx, ny):
        # generate points
        x, y = np.meshgrid(np.linspace(0, self.l, nx), np.linspace(0, self.l, ny))
        x = x.flatten()
        y = y.flatten()
        return x, y

    def boundary_conditions(self, x, y):
        sigma_x = -(self.q0 / self.l) * (x - (2 * y))
        sigma_y = (self.q0 / self.l) * x
        tau_xy = (self.q0 / self.l) * y
        return sigma_x, sigma_y, tau_xy
