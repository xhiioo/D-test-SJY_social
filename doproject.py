# doproject.py

from sampleset import FeatureSpace, SampleSet
from lambdas import Lambdas
from featuretype import Constant, Product, Lam, Threshold
import numpy as np

class Project:
    def __init__(self, featurespace, featuretype, sampleset):
        self.grids = featurespace
        self.featuretypes = featuretype
        self.sample = sampleset


    def eval(self):
        for grid in self.grids:
            for feature in self.featuretypes:
                return feature.eval(grid)


# class GridAndFeature:
#     def __init__(self, grid, features):
#         self.grid = grid
#         self.linear = []
#         self.threshold = []
#         self.hinge = []
#         self.revhinge = []
#
#     def computecell(self, val):
#         pass
#
#     def computegrid(self):
#         pass
#


if __name__ == '__main__':
    fname = r'D:\test\SJY\asc'
    csvfile = r'D:\test\SJY\with9factors\settlements_samplePredictions.csv'
    lambdafile = r'D:\test\SJY\with9factors\settlements.lambdas'
    fs = FeatureSpace(fname)
    ss = SampleSet(csvfile)
    l = Lambdas(lambdafile)
    l.parselambdafile()

    ss.getbgvalues(fs)
    dt = ss.getfeatures()
    for nm in dt.columns:
        for v in l.lambdas:
            if nm in v.name:
                print(nm, v)

    res = []
    for v in l.lambdas:
        if isinstance(v, Product):
            g1, g2 = v.name
            pp = v.eval(ss.csv[g1], ss.csv[g2]) * v.lam
            res.append(pp)
        elif isinstance(v, Lam):
            g1 = v.name
            pp = v.eval(ss.csv[g1]) * v.lam
            res.append(pp)

    rr = sum(res)
    p = [(ll.name, ll.value) for ll in l.lambdas if isinstance(ll, Constant)]
    rrr = np.exp(rr - p[0][1]) / p[1][1]

    import matplotlib.pyplot as plt

    plt.scatter(rrr, ss.csv['Raw prediction'])
    plt.xlim((-0.0005, 0.006))
    plt.ylim((-0.0005, 0.002))
    plt.show()

    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(ss.X, ss.Y, rrr, marker=1)
    # ax.scatter(ss.X, ss.Y, ss['Raw prediction'], marker=2)

    import pandas as pd

    res = {}
    itv = {}
    fig, axs = plt.subplots(4,4)
    for ly, ax in zip(fs.layers.items(), axs.flat):
        nm = ly[0]
        lyr = ly[1]
        arr = lyr.array()
        arr = arr[arr > -9999]
        dt = ss.csv[nm]
        v = [ll.threshold for ll in l.lambdas if nm == ll.name and isinstance(ll, Threshold)]
        v = sorted(v)
        itv[nm] = v
        tb1 = pd.value_counts(pd.cut(arr, v))
        tb1 = tb1.iloc[np.argsort(tb1.index)]
        res1 = np.cumsum(tb1/sum(tb1))
        tb2 = pd.value_counts(pd.cut(dt, v))
        tb2 = tb2.iloc[np.argsort(tb2.index)]
        res2 = np.cumsum(tb2/sum(tb2))
        res[nm] = (res1, res2)
        ax.hist([res1, res2])
        ax.scatter(v, np.ones(len(v)) * 1.1)





def interpolate(x=[], y=[], xx=0):
    for i in range(len(x)):
        if xx < x[i]:
            break
    if i == 0:
        return y[0]
    elif i == len(x):
        return y[len(x) - 1]
    else:
        return y[i-1] + (y[i]-y[i-1]) * (xx-x[i-1])/(x[i]-x[i-1])







