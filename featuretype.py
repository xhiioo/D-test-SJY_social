# featuretype.py
import numpy as np

class AbstractFeature:
    __dict__ = ['samplexpectation', 'sampledeviation', 'featurexpectation', 'featuredeviation', 'lam', 'beta']
    def __init__(self, **kwargs):
        self.samplexpectation = kwargs.get('samplexpectation', 0)
        self.sampledeviation = kwargs.get('sampledeviation', 0)
        self.featurexpectation = kwargs.get('featurexpectation', 0)
        self.featuredeviation = kwargs.get('featuredeviation', 0)
        self.lam = kwargs.get('lam', 0)
        self.beta = kwargs.get('beta', 1.0)
        self.lastChange = -1

    def eval(self, val=0):
        pass

    def getSampleExpectation(self):
        return self.samplexpectation

    def getSampleDeviation(self):
        return self.sampledeviation

    def getExpectation(self):
        return self.featurexpectation

    def getDeviation(self):
        return self.featuredeviation

    def getLambda(self):
        return self.lam

    def getBeta(self):
        return self.beta

    def increaseLambda(self, x):
        self.lam += x

    def isGenerated(self):
        return isinstance(self, Threshold)


class Linear(AbstractFeature):
    '''Linear: Parameters(name, lam, min, max, 'samplexpectation',\
     'sampledeviation', 'featurexpectation', 'featuredeviation', 'lam', 'beta')'''
    def __init__(self, name='', lam=0, min=0, max=1, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.lam = float(lam)
        self.min = float(min)
        self.max = float(max)
        self.scale = np.subtract(self.max, self.min)

    def eval(self, val=0):
        return (np.subtract(val, self.min))/self.scale

    def __repr__(self):
        return '<{0:20}: {1:<15} max: {2:<6.5} min: {3:<6.5}>'.format('Linear Feature',
                                                                  self.name,
                                                                  self.max,
                                                                  self.min)
    def __hash__(self):
        return hash((self.name, type(self)))

class Product(AbstractFeature):
    def __init__(self, grid1='', grid2='', lam=0, min=0, max=0, **kwargs):
        super().__init__(**kwargs)
        self.name = (grid1, grid2)
        self.lam = float(lam)
        self.min = float(min)
        self.max = float(max)
        self.scale = np.subtract(self.max, self.min)

    def eval(self, val1=0, val2=0):
        return np.subtract(np.multiply(val1, val2), self.min)/self.scale

    def __repr__(self):
        return '<{0:<25}: {1:<30}>'.format('Product Feature',
                                           str.join(' & ', self.name))

    def  __hash__(self):
        return hash(self.name)

class Threshold(AbstractFeature):
    def __init__(self, threshold=0, name='', lam=0, min=0, max=0, thrnum=-1, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.lam = float(lam)
        self.min = float(min)
        self.max = float(max)
        self.threshold = float(threshold)
        self.thrnum = thrnum

    def eval(self, val=0):
        return np.where(np.array(val) > self.threshold, 1, 0)

    def __repr__(self):
        return '<{0:<20}: {1:<15} on {2:<6.5f}>'.format('Threshold Feature',
                                                   self.name,
                                                   self.threshold)

    def __hash__(self):
        return hash((self.name, self.threshold))

    def __eq__(self, other):
        try:
            res = (self.name == other.name) & (self.threshold == other.threshold)
        except:
            return False
        return res

class Hinge(Linear):
    def eval(self, val=0):
        val = np.array(val)
        res = (val - self.min) / self.scale
        res[val < self.min] = 0
        res[val > self.max] = 1
        return res

    def __repr__(self):
        return '<{0:20}: {1:<15} on {2:<6.5f} and {3:<6.5f}>'.format('Hinge Feature',
                                                                   self.name,
                                                                   self.min,
                                                                   self.max)


class Revhinge(Hinge):
    def eval(self, val=0):
        return -super().eval(val)

    def __repr__(self):
        return '<{0:<20}: {1:<15} on {2:<6.5f} and {3:<6.5f}>'.format('Revhinge Feature',
                                                                   self.name,
                                                                   self.min,
                                                                   self.max)

class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = float(value)

    def __repr__(self):
        return '<{0:<25}: {1:<25}: {2:<6.5f}>'.format('Constant Number',
                                                     self.name,
                                                     self.value)



