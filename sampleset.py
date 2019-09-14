# sampleset.py
# include FeatureSpace and SampleSet

from readcsv import txt2dfm
from os import listdir
from os.path import splitext, join
from feature import Feature
from featuretype import AbstractFeature, Threshold
import pandas as pd
import numpy as np
from myraster import lazyproperty

eps = 0.0000001

def precision(f=0.0):
    if (f == 0) | (f is np.nan):
        return 1
    f = np.abs(f)
    firstSig = np.floor(np.log(f)/np.log(10))
    currentPower = np.power(10.0, firstSig)
    lastTwoDigits = 0
    for i in range(1, 7):
        try:
            currentDigit = int(f/currentPower)
        except:
            return 1
        lastTwoDigits = (lastTwoDigits*10+currentDigit)%100
        if i >= 2:
            if (lastTwoDigits == 00) | (lastTwoDigits == 1)\
                    | (lastTwoDigits == 99) | (lastTwoDigits == 98):
                return currentPower*50.0
        f -= currentDigit*currentPower
        currentPower /= 10.0
    return currentPower*5

class SampleInfo:
    def __init__(self, avg=0.0, std=0.0, min=0.0, max=1.0, numsample=1):
        self.avg = avg
        self.std = std
        self.min = min
        self.max = max
        self.sample_cnt = numsample

class Interval:
    def __init__(self, *args, **kwargs):
        for fun in [self._init1, self._init2, self._init3, self._init4]:
            try:
                fun(*args, **kwargs)
                break
            except:
                continue

    def _init1(self, *args, **kwargs):
        low = kwargs['low']
        high = kwargs['high']
        self.low = low
        self.high = high

    def _init2(self, *args, **kwargs):
        f = kwargs['sampleinfo']
        beta = kwargs['beta']
        self.low = f.avg - beta/np.sqrt(f.sample_cnt)*f.std
        self.high = f.avg + beta/np.sqrt(f.sample_cnt)*f.std

    def _init3(self, *args, **kwargs):
        interval1 = kwargs['interval1']
        interval2 = kwargs['interval2']
        if interval2.low < 0:
            self.low = np.inf
            self.high = -np.inf
        else:
            self.low = interval1.low/interval2.high
            self.high = interval1.high/interval2.low

    def _init4(self, *args, **kwargs):
        sampleinfo1 = kwargs['sampleinfo1']
        sampleinfo2 = kwargs['sampleinfo2']
        beta = kwargs['beta']
        iterv1 = Interval(sampleinfo=sampleinfo1, beta=beta)
        iterv2 = Interval(sampleinfo=sampleinfo2, beta=beta)
        self._init3(interval1 = iterv1, interval2 = iterv2)

    def getMid(self):
        return 0.5*(self.low + self.high)

    def getDev(self):
        return 0.5*(self.high - self.low)


class FeatureSpace:
    def __init__(self, file=''):
        all = listdir(file)
        fs = [f for f in all if f.endswith('.asc')]
        # fs.append('elevation.asc')

        self.layers = {}
        for fn in fs:
            name = splitext(fn)[0]
            fullname = join(file, fn)
            self.layers[name] = Feature(fullname)



    def __repr__(self):
        for k, v in self.layers.items():
            print('{0}:{1} {2}'.format(k, ' '*(25-len(k)), v))
        return ''

    def getValueSet(self, x, y):
        d = {}
        for k, v in self.layers.items():
            print("Reading ", k, ': ')
            d[k] = v.getValueList(x, y)
        return d

    def getlayervalues(self, colrowtuple):
        x, y = zip(*colrowtuple)
        df = pd.DataFrame({'col':x, 'row':y})
        for k, v in self.layers.items():
            df[k] = v.getValueList(colrowtuple)
        return df



class SampleSet:
    def __init__(self, csvfile=''):
        self.csv = txt2dfm(csvfile)

    def __repr__(self):
        print(self.csv)
        return ''

    def __getattr__(self, item):
        return getattr(self.csv, item)

    def __getitem__(self, item):
        return self.csv.__getitem__(item)

    def __setitem__(self, key, value):
        self.csv.__setitem__(key, value)

    def __delitem__(self, key):
        self.csv.__delitem__(key)

    def getbgvalues(self, fs):
        d = fs.getValueSet(self.csv.X, self.csv.Y)
        for k, v in d.items():
            self.csv[k] = v

    def getfeatures(self):
        return self.csv.iloc[:,6:]



class ThrFeatureGenerator:
    def __init__(self, s='', d1=pd.Series, d2=pd.Series):
        self.name = s
        self.samples = d1
        self.points = d2
        self.vals = self.samples.append(self.points).sort_values().reset_index(drop = True)
        self.prec = min([precision(v) for v in self.vals])
        self.beta = 1.0
        self.min = np.nanmin(self.vals)
        self.max = np.nanmax(self.vals)

    @lazyproperty
    def thr(self):
        thr = []
        t = min(self.vals)
        thr.append(t)
        for s in self.vals:
            if s > t + self.prec:
                thr.append((t+s)/2.0)
                t = s
        self.lambdas = np.zeros(len(thr))
        return thr

    def setThrExpectation(self):
        numSample = len(self.samples)
        sum1 = numSample - self.samples.sort_values().searchsorted(self.thr)
        avg = sum1/numSample
        std = np.sqrt((sum1-numSample * avg * avg)/(numSample-1))
        self.samplexpectations = avg
        self.sampledeviation = std
        l = []
        biasinfo = SampleInfo(1.0, 0.0, 1.0, 1.0, numSample)
        for a, s in zip(avg, std):
            sampleinfo = SampleInfo(a, s, 0.0, 1.0, 115)
            res = Interval(sampleinfo1 = sampleinfo, sampleinfo2 = biasinfo, beta = self.beta)
            rr = res.getDev()
            l.append(rr)
        self.sampledeviation = np.array(l)
        numPoints = len(self.points)
        sum1 = numPoints - self.points.sort_values().searchsorted(self.thr)
        avg = sum1/numPoints
        std = np.sqrt((sum1-numPoints * avg * avg)/(numSample-1))
        self.featurexpectations = avg
        self.featuredeviation = std

    def exportFeature(self, num):
        return Threshold(threshold=self.thr[num], name=self.name,
                         lam=self.lambdas[num], min=self.min, max=self.max, thrnum=num,
                         samplexpectation=self.samplexpectations[num],
                         sampledeviation=self.sampledeviation[num],
                         featurexpectation=self.featurexpectations[num])


    def updateFeatureExpectations(self, X):
        arg = np.argsort(self.points).tolist()
        ind = self.points.sort_values().searchsorted(self.thr)
        try:
            vals = np.sum(X.density) - (X.density.iloc[arg]).cumsum()
        except:
            print('error')
        fe = self.featurexpectations
        self.featurexpectations = vals.reset_index(drop = True)[ind].reset_index(drop=True)/X.densityNormalizer






if __name__ == '__main__':
    # fname = r'D:\test\SJY\asc'
    # csvfile = r'D:\test\SJY\shp\shp\village.csv'
    fname = r'E:\xxx\MaxEnt\layers'
    csvfile = r'E:\xxx\MaxEnt\samples\bradypus_copy.csv'
    fs = FeatureSpace(fname)
    sampleset = SampleSet(csvfile)

    sampleset.getbgvalues(fs)
    bglist = fs.layers['h_dem'].getRDsamples(10000)
    bgset = fs.getlayervalues(bglist)






