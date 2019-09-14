import gdal
import numpy as np

class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

class Raster:
    def __init__(self, r=''):
        if r != '':
            self.raster = gdal.Open(r)
            self.xMin, self.xr, self.xOff, self.yMin, self.yOff, self.yr \
                = self.raster.GetGeoTransform()
            self.xSize = self.raster.RasterXSize
            self.ySize = self.raster.RasterYSize
            self._band = self.raster.GetRasterBand(1)
            self.min, self.max = self._band.ComputeRasterMinMax()
            self.NODATA = self._band.GetNoDataValue()
        else:
            self.raster = ''

    @lazyproperty
    def array(self):
        arr = self._band.ReadAsArray()
        try:
            arr[arr==self.NODATA] = np.nan
        except ValueError:
            arr = arr.astype(np.float)
            arr[arr == self.NODATA] = np.nan
        return arr

    def geotransform(self):
        if self.raster != '':
            return self.raster.GetGeoTransform()
        return ''

    def xytorowcol(self, x, y):
        x = np.array(x)
        y = np.array(y)
        cols = ((x - self.xMin - self.xOff) // self.xr).astype(np.int)
        rows = ((y - self.yMin - self.yOff) // self.yr).astype(np.int)
        return(rows, cols)

    def getvaluelst(self, x, y):
        rows, cols = self.xytorowcol(x, y)
        return self.array[rows, cols]

    def createXYgrid(self, *args):
        if len(args) == 2:
            xr = args[0]
            yr = args[1]
        xr = self.xr
        yr = self.yr
        x = np.arange(self.xMin, self.xMin + self.xSize * xr, step=xr)
        y = np.arange(self.yMin, self.yMin + self.ySize * yr, step=yr)
        x, y = np.meshgrid(x, y)
        return np.array([x,y])

if __name__ != 'main':
    r = Raster(r"D:\test\SJY\asc\elevation.asc")
    r.geotransform()
    r.getvaluelst([], [])
    r.createXYgrid()



