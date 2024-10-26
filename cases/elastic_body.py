from abc import abstractmethod


class ElasticBody:
    def __init__(self, e, nu):
        self.e = e
        self.nu = nu

    @abstractmethod
    def discretize(self, nx, ny):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def boundary_conditions(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses.")
