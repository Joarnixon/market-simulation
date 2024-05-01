import numpy as np
from numpy import eye, array, where, clip, corrcoef, sin, pi
from math import sqrt
from scipy.stats import norm
from numpy.random import multivariate_normal


class Characteristic:
    def __init__(self, name, data_type='uniform', p=0):
        self.name = name
        self.data_type = data_type
        self.p = p

    def __eq__(self, other: str):
        return self.name == other


class CharacteristicArray(np.ndarray):
    def __new__(cls, data, names):
        obj = np.asarray(data).view(cls)
        return obj

    def __init__(self, data, names):
        super().__init__()
        self.names = names

    def __mul__(self, other):
        data = super(CharacteristicArray, self).__mul__(other)
        return data

    def get(self, name: str):
        idx = self.names.index(name)
        return self[idx]

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, 'names', None)


class CharacteristicsGenerator:
    def __init__(self):
        self.characteristics = []
        self.characteristics_names = []
        self.correlation_matrix = []
        self.bernoulli_adjust_poly = None
        self.bernoulli_adjust_model = None

    def complete_init(self):
        self.correlation_matrix = 2 * sin(pi/6 * array(self.correlation_matrix))
        self.characteristics_names = [char.name for char in self.characteristics]

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)
        self._create_correlation_matrix()

    def update_correlation(self, char1, char2, correlation):
        index1 = self.characteristics.index(char1)
        index2 = self.characteristics.index(char2)
        # TODO: If using any bernoulli values then correlation is ruined
        if (self.characteristics[index1].data_type == 'bernoulli') or (self.characteristics[index2].data_type == 'bernoulli'):
            criteria = max(self.characteristics[index1].p, self.characteristics[index2].p)
            correlation = clip(correlation / sqrt(3) * (1 / (sqrt(criteria * (1 - criteria)))) * (1 / (1 - (criteria ** 2) / 2 + 3 * (criteria**3) / 5)), -1, 1)
        self.correlation_matrix[index1][index2] = correlation
        self.correlation_matrix[index2][index1] = correlation

    def _create_correlation_matrix(self):
        n = len(self.characteristics)
        self.correlation_matrix = eye(n)

    def __call__(self):
        chars = array(norm.cdf(multivariate_normal(mean=[0]*len(characteristics), cov=self.correlation_matrix)))
        ind = [char.data_type == 'bernoulli' for char in self.characteristics]
        ps = array([char.p for char in self.characteristics])
        chars[ind] = where(chars[ind] > ps[ind], 1, 0)
        return CharacteristicArray(chars, self.characteristics_names)


characteristics_generator = CharacteristicsGenerator()

characteristics = [
    ['sex', 'bernoulli', 0.5],
    ['strength'],
    ['workaholic'],
    ['plainness'],
    ['greed']
]

[characteristics_generator.add_characteristic(Characteristic(*char)) for char in characteristics]

# correlation > 0 => bigger char1 then bigger char2
# correlation < 0 => bigger char1 then smaller char2
# correlation is in [0, 1]
# By default is 0

characteristics_correlation = [
    ['sex', 'strength', 0.5],
    ['sex', 'workaholic', -0.2],
    ['sex', 'plainness', 0.2],
    ['sex', 'greed', 0.4],
    ['workaholic', 'plainness', -0.5],
    ['strength', 'workaholic', 0.3]
]

[characteristics_generator.update_correlation(*corr) for corr in characteristics_correlation]
characteristics_generator.complete_init()

#
# Testing
# #
# points1 = characteristics_generator()
# points2 = characteristics_generator()

# print(points1)
# print(points2)
# print(points1 * points2)
# print(points1 @ characteristics_generator.correlation_matrix @ points2)
# print(points1.get('strength'))
# print(characteristics_generator.correlation_matrix)
# print(where((0.1 < corrcoef(array(points).T)) + (corrcoef(array(points).T) < -0.1), corrcoef(array(points).T), 0))
# print(points[0])