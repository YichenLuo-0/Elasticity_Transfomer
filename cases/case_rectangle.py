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
        return np.ones_like(x, dtype=bool)

    @overrides
    def boundary_conditions(self, x, y):
        sigma_x = -(self.q0 / self.l) * (x - (2 * y))
        sigma_y = (self.q0 / self.l) * x
        tau_xy = (self.q0 / self.l) * y
        return sigma_x, sigma_y, tau_xy
