from abc import abstractmethod

import numpy as np


class ElasticBody:
    def __init__(self, e, nu, x_min, x_max, y_min, y_max):
        self.e = e
        self.nu = nu
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def geometry(self, nx, ny):
        x, y = np.meshgrid(np.linspace(self.x_min, self.x_max, nx), np.linspace(self.y_min, self.y_max, ny))
        x = x.flatten()
        y = y.flatten()
        mask = self.geo_filter(x, y)
        return x[mask], y[mask]

    def boundary_conditions(self, x, y):
        bc = np.zeros([x.shape[0], 7])

        # generate boundary conditions
        # Boundary condition encoding format: [l, m, tx, ty, u, v, mask]
        # l, m: normal vector
        # tx, ty: For force boundary, tx, ty are the components of the force
        # u, v: For displacement boundary, u, v are the components of the displacement
        # mask: 0-internal point, 1-displacement boundary, 2-force boundary
        for i in range(x.shape[0]):
            x_idx = x[i]
            y_idx = y[i]
            bc[i] = self.bc_filter(x_idx, y_idx)
        return bc

    @abstractmethod
    def geo_filter(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def bc_filter(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def ground_truth(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses.")
