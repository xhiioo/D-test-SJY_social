import numpy as np
import pandas as pd


def txt2dfm(txt = '', header = True):
    with open(txt, 'r') as f:
        ls = f.readlines()

    lss = [l.split('\n')[0].split(',') for l in ls]
    if header:
        nms = lss[0]
        df = pd.DataFrame(lss[1:], columns=nms, dtype=np.float)
        row, col = df.shape
    else:
        nms =[chr(i) for i in range(97, 97+len(lss[0]))]
        df = pd.DataFrame(lss[0:], columns=nms, dtype=np.float)
        row, col = df.shape
    print("Rows: {0}, Cols: {1}.\n"
          "Column Names: {2}".format(row, col, nms))
    return df


if __name__ == 'main':
    df = txt2dfm(r"D:\test\SJY\with9factors\settlements_samplePredictions.csv")
    # scatter3d(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 2])






