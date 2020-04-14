from collections import namedtuple
import numpy as np


Size = namedtuple('Size', ['x', 'y'])
SIZE = Size(10, 10)
ENERGY_HL = 30
ENERGY_LL = 10

energy_field =


class Gene:
    def __init__(self):
        self.code = np.zeros((1, 4))


class Genome:
    def __init__(self):
        pass


class Tree:
    def __init__(self):
        pass


class Environment:
    def __init__(self):
        pass


class EnergyField:
    def __init__(self, size_x, size_y):
        # create field
        self.field = np.zeros((size_x, size_y))
        # add energy distribution
        self.field += np.expand_dims(np.flip(
            np.linspace(start=ENERGY_LL, stop=ENERGY_HL, num=SIZE.y, dtype=np.uint8)), axis=1)

    def get_field(self):
        return self.field

