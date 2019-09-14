# feature.py
from myraster import Raster
import numpy as np
from os.path import basename, splitext

class Feature:
    def __init__(self, fname):
        name = basename(fname)
        name = splitext(name)[0]
        self.name = name
        self._raster = Raster(fname)
        self.min = self._raster.min
        self.max = self._raster.max

    def array(self):
        return self._raster.array

    def eval(self, x, y):
        return (self.array[x, y] - self.min)/(self.max - self.min)

    def getValueList(self, *args):
        if len(args) == 2:
            return self._raster.getvaluelst(args[0], args[1])
        if len(args) == 1:
            x = [v[0] for v in args[0]]
            y = [v[1] for v in args[0]]
            return self.array()[x, y]

    def getRDsamples(self, n):
        arr = self.array().copy()
        arr = np.where(np.isnan(arr), 0, 1)
        cnt = 0
        sam = []
        while(cnt < n):
            c = np.random.randint(0, self._raster.xSize, n)
            r = np.random.randint(0, self._raster.ySize, n)
            l = [(cc, rr) for cc, rr in zip(r, c) if arr[cc, rr] == 1]
            cnt += len(l)
            sam.extend(l)
        return sam[0:n]



def main():
    fname = r'E:\xxx\MaxEnt\layers\h_dem.asc'
    # fname = 'D:\\test\\SJY\\asc\\elevation.asc'
    r = Feature(fname)
    print(r.max, r.min)
    arr = r.array
    print(arr)
    p = r.eval(1000, 1000)

if __name__ == '__main__':
    main()