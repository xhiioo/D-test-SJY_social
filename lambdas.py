# lambdas.py
import re
from featuretype import *
from collections import defaultdict

class Lambdas:
    patterns = [
        '([a-z_]*), ([-]?[0-9.]*), ([-]?[0-9.]*), ([-]?[0-9.]*)\n',
        '(.*)\*(.*), (.*), (.*), (.*)',
        '\((.*)<(.*)\), (.*), (.*), (.*)\n',
        '`([a-z_]*), (.*), (.*), (.*)\n',
        '\'([a-z_]*), (.*), (.*), (.*)\n',
        '([a-zA-Z_]*), ([-]?[0-9.]*)\n'
    ]

    formats = [
        Linear,
        Product,
        Threshold,
        Revhinge,
        Hinge,
        Constant
    ]

    def __init__(self, lambdafile=''):
        self.file = lambdafile
        self.lambdas = []

    def eval(self, s=''):
        for pattern, t in zip(self.patterns, self.formats):
            m = re.match(pattern, s)
            if m is not None:
                return t(*m.groups())


    def parselambdafile(self):
        with open(self.file) as fp:
            line = fp.readline()
            while line != '':
                try:
                    t = self.eval(line)
                    self.lambdas.append(t)
                except:
                    line = fp.readline()
                line = fp.readline()


# def main():
#     lambdafile = r'D:\test\SJY\with9factors\settlements.lambdas'
#     l = Lambdas(patterns)
#     with open(lambdafile) as fp:
#         line = fp.readline()
#         while line != '':
#             l.eval(line)
#             line = fp.readline()
#
#     for _ in range(0, 5):
#         l.Print(_)
#         print('\n')

if __name__ == '__main__':
    lambdafile = r'D:\test\SJY\with9factors\settlements.lambdas'
    l = Lambdas(lambdafile)
    l.parselambdafile()

    # for _ in range(0, 5):
    #     l.Print(_)
    #     print('\n')

