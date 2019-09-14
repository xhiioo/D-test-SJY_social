# test.py
from myraster import Raster
from readcsv import txt2dfm
import numpy as np
import scipy.stats
import lambdas

xs = np.linspace(0, 1, 100)


def makefeature(fname = '', dfm = None):
    rst = Raster(fname)
    arr = rst.getvaluemat()
    dt = rst.getvaluelst(dfm.x, dfm.y)

    # scale
    arr = arr[arr != -9999]
    arr_min = arr.min()
    arr_max = arr.max()
    arr_s = (arr - arr_min)/(arr_max - arr_min)
    dt_s = (dt - arr_min)/(arr_max - arr_min)

    k1 = scipy.stats.gaussian_kde(arr_s)
    k2 = scipy.stats.gaussian_kde(dt_s)

    p1 = k1.evaluate(xs)
    p2 = k2.evaluate(xs)

    return p1, p2



def main():
    fname = r'D:\test\SJY\asc\elevation.asc'
    csvfile = r'D:\test\SJY\shp\shp\village.csv'

    dem = Raster(fname)
    csv = txt2dfm(csvfile)

    arr = dem.getvaluemat()
    arr = arr[arr != -9999]
    data = dem.getvaluelst(csv.x, csv.y)

    k1 = scipy.stats.gaussian_kde(arr)
    k2 = scipy.stats.gaussian_kde(data)

    xs = np.linspace(min(arr.min(), data.min()), max(arr.max(), data.max()), 100)
    ## kernel estimate for prior and observe data
    p1 = k1.evaluate(xs)
    p2 = k2.evaluate(xs)

    ## normal and gamma estimate
    mu = arr.mean()
    sigma2 = arr.var()
    np1 = scipy.stats.norm.pdf(xs, mu, np.sqrt(sigma2))


    lambdafile = r'D:\test\SJY\with9factors\settlements.lambdas'
    l = lambdas.Lambdas(lambdas.patterns)
    with open(lambdafile) as fp:
        line = fp.readline()
        while line != '':
            l.eval(line)
            line = fp.readline()



if __name__ == '__main__':
    pass


