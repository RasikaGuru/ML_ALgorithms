import pandas as pd
import numpy as np
import math

def eucledian_distance(a1,a2,b1,b2):
    value1 = pow(a1-b1,2)
    value2 = pow(a2-b2,2)
    dist = pow(value1+value2, 0.5)
    return dist


if __name__ == "__main__":
    df = pd.read_csv('clusters.txt', header=None)

    centroidlist = (df.sample(n=3)).values.T.tolist()
    for index,row in df.iterrows():
        for i in range(len(centroidlist)):

            a1 = df[index][0]
            a2 = centroidlist[i][0]
            b1 = df[index][1]
            b2 = centroidlist[i][1]
            print a1
            print a2
            distance = eucledian_distance(a1,a2,b1,b2)
            print distance




