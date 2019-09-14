## sequential.py
from sampleset import FeatureSpace, SampleSet, ThrFeatureGenerator, SampleInfo, Interval
from featuretype import AbstractFeature, Linear
import numpy as np

EPS = 1e-6
BIASEDFEATURE = 1

def goodAlpha(h=AbstractFeature()):
    N1 = h.getSampleExpectation()
    W1 = h.getExpectation()
    W0 = 1 - W1
    N0 = 1 - N1
    lam = h.getLambda()
    beta1 = h.getSampleDeviation()
    try:
        with np.errstate(divide='raise', invalid='raise'):
            alpha1 = np.log((N1 - beta1) * W0 / ((N0 + beta1) * W1))
            alpha2 = np.log((N1 + beta1) * W0 / ((N0 - beta1) * W1))
    except:
        alpha1 = np.nan
        alpha2 = np.nan
    if (~((N1 - beta1 > EPS) & (alpha1 + lam > 0.0))
            &
            ~((N1 - beta1 > EPS) & (alpha1 + lam < 0.0))):
        alpha2 = -lam
    return alpha2


def reduceAlpha(alpha = 0, iteration=0):
    if iteration < 10:
        return alpha/50
    elif iteration < 20:
        return alpha/10
    elif iteration < 50:
        return alpha/3
    return alpha


def deltaLossBound(h=AbstractFeature()):
    N1 = h.getSampleExpectation()
    if N1 == -1:
        return 0
    W1 = h.getExpectation()
    W0 = 1 - W1
    N0 = 1 - N1
    lam = h.getLambda()
    alpha = goodAlpha(h)
    beta1 = h.getSampleDeviation()
    if alpha is np.inf:
        return 0
    bound = -N1 * alpha \
            + np.log(W0 + W1 * np.exp(alpha)) \
            + beta1 * (np.abs(lam + alpha) - np.abs(lam))
    return 0 if bound is np.inf else bound


class sequential:
    def __init__(self, fname='', csvfile='', backgroundnum=10000):
        self.fs = FeatureSpace(fname)
        self.featurename = list(self.fs.layers.keys())
        self.sampleset = SampleSet(csvfile)
        self.sampleset.getbgvalues(self.fs)
        bglist = self.fs.layers['h_dem'].getRDsamples(backgroundnum)
        self.bgset = self.fs.getlayervalues(bglist)

        self.density = []
        self.linearPredictor = np.zeros(backgroundnum)
        self.numFeatures = len(self.fs.layers.keys())
        self.features = []
        self.featureGenerators = []
        self.linearPredictorNormalizer = 0.0
        self.densityNormalizer = self.bgset.shape[0]
        self.entropy = -1.0
        self.reg = 0
        self.iteration = -1
        self.activeLayer = {'ecoreg', 'h_dem', 'tmp6190_ann'}

    def isActive(self, nm):
        return nm in self.activeLayer

    def featuresToUpdate(self):
        toUpdate = []
        dlb = []
        for feature in self.features:
            last = feature.lastChange
            dlb.append(deltaLossBound(feature))
        orderedDlb = np.argsort(dlb)
        for n in orderedDlb:
            if not self.features[n].isGenerated():
                toUpdate.append(self.features[n])
        return toUpdate

    def increaseLambda(self, f=AbstractFeature(), alpha=0.0, toUpdate=[]):
        self.reg += np.abs(f.getLambda() + alpha) - np.abs(f.getLambda()*f.getSampleDeviation())
        if alpha == 0:
            return None
        f.increaseLambda(alpha)
        for fg in self.featureGenerators:
            if f.name == fg.name:
                fg.lambdas[f.thrnum] += alpha
                break
        self.linearPredictor += f.eval(self.bgset[f.name]) * alpha
        self.linearPredictorNormalizer = max(max(self.linearPredictor), self.linearPredictorNormalizer)
        for feature in toUpdate:
            # Lastchange = iteration. to be complete
            feature.lastExpectationUpdate = 0
        self.setDensity(toUpdate)
        return self.getLoss()



    def doSequentialUpdate(self, feature=AbstractFeature(), iteration=0):
        newLoss = self.getLoss()
        oldLambda = feature.getLambda()
        feature.lastChange = self.iteration
        toUpdate = self.featuresToUpdate()
        dlb = deltaLossBound(feature)

        if (True):
            alpha = goodAlpha(feature)
            alpha = reduceAlpha(alpha, iteration)
            newLoss = self.increaseLambda(feature, alpha, toUpdate)
        else:
            pass
        return newLoss

    def feature2Generator(self, featureName=''):
        return ThrFeatureGenerator(featureName, self.sampleset[featureName], self.bgset[featureName])

    def getLoss(self):
        sum = 0
        for feature in self.features:
            sum += feature.lam * feature.getSampleExpectation()
        try:
            with np.errstate(divide = 'raise'):
                res = -sum + self.linearPredictorNormalizer + np.log(self.densityNormalizer) + self.reg
        except:
            print("error: ", sum)
            return None
        return res

    def getN1(self, *args):
        if args:
            pass
        else:
            pass

    def linearPredictor(self, sample):
        pass

    def setDensity(self, toUpdate=[]):
        self.density = BIASEDFEATURE * np.exp(self.linearPredictor - self.linearPredictorNormalizer)
        density_sum = np.zeros(len(toUpdate))
        for i, feature in enumerate(toUpdate):
            density_sum[i] = np.sum(feature.eval(self.bgset[feature.name]) * self.density)
        self.densityNormalizer = np.sum(self.density)
        for i, feature in enumerate(toUpdate):
            feature.expectation = density_sum[i] / self.densityNormalizer
        for featureGenerator in self.featureGenerators:
            featureGenerator.updateFeatureExpectations(self)

    def getDensity(self, sample):
        pass

    def getBestFeature(self):
        bestlb = np.inf
        bestFeature = None
        for feature in self.featureGenerators:
            for num in range(len(feature.thr)):
                ft = feature.exportFeature(num)
                bound = deltaLossBound(ft)
                if bound < bestlb:
                    bestlb = bound
                    bestFeature = ft
        eq = False
        for ft in self.features:
            if ft == bestFeature:
                eq = True
                break
        if not eq:
            self.features.append(bestFeature)
        return bestFeature


    def newDensity(self):
        pass

    def scaledBiasDist(self, biasDistFeature):
        pass

    def setFeatures(self):
        for nm in self.featurename:
            if self.isActive(nm):
                feature = Linear(nm, 0, self.fs.layers[nm].min, self.fs.layers[nm].max)
                fInfo =  SampleInfo(feature.eval(self.sampleset[nm]).mean(),
                                    feature.eval(self.sampleset[nm]).std(),
                                    feature.eval(self.sampleset[nm]).min(),
                                    feature.eval(self.sampleset[nm]).max())
                biasInfo = SampleInfo(1.0, 0.0, 1.0, 1.0, self.sampleset.shape[0])
                fInterval = Interval(sampleinfo1 = fInfo, sampleinfo2 = biasInfo, beta = 0.05)
                feature.samplexpectation = fInterval.getMid()
                feature.sampledeviation = fInterval.getDev()
                self.features.append(feature)

    def setFeatureGenerators(self):
        for feature in self.features:
            nm = feature.name
            featureGenerator = ThrFeatureGenerator(nm, self.sampleset[nm], self.bgset[nm])
            featureGenerator.setThrExpectation()
            self.featureGenerators.append(featureGenerator)

    def setBiasDiv(self):
        pass

    def setLinearPredictor(self):
        for feature in self.features:
            self.linearPredictor += feature.eval(self.bgset[feature.name]) * feature.getLambda()
        self.linearPredictorNormalizer = np.min(self.linearPredictor)

    def setBiasDist(self):
        pass

    def goodAlpha(self):
        pass

    def getEntropy(self):
        pass

    def run(self):
        newLoss = self.getLoss()
        for iteration in range(70):
            oldLoss = newLoss
            if False:
                newLoss = doParalleUpdateFrequency(-1)
            bestFeature = self.getBestFeature()
            if (bestFeature==None):
                break
            newLoss = self.doSequentialUpdate(bestFeature, iteration)

    def predict(self):
        pointsres = np.zeros(self.bgset.shape[0])
        sampleres = np.zeros(self.sampleset.shape[0])
        for feature in self.features:
            sampleres += feature.eval(self.sampleset[feature.name]) * feature.lam
            pointsres += feature.eval(self.bgset[feature.name]) * feature.lam
        return sampleres, pointsres


if __name__ == "__main__":
    fname = r'E:\xxx\MaxEnt\layers'
    csvfile = r'E:\xxx\MaxEnt\samples\bradypus_copy.csv'

    s = sequential(fname, csvfile, 10000)
    s.setFeatures()
    s.setFeatureGenerators()
    s.setBiasDiv()
    s.setLinearPredictor()
    s.setBiasDist()

    s.run()


