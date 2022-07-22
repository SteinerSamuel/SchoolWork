import numpy as np
from scipy.spatial.distance import cosine, euclidean
import itertools
from numpy.linalg import norm 
import matplotlib.pyplot as plt
import math

n=[100, 1000]
k=range(1,101)

def calcDiff_euc(dp, _n):
    pairs = itertools.combinations(dp, 2)
    diff=[]
    for t in pairs:
        diff.append(norm(t[0]-t[1]))
    return diff


def calcDiff_cos(dp, _n):
    pairs = itertools.combinations(dp, 2)
    diff=[]
    for t in pairs:
        diff.append(cosine(t[0], t[1]))
    return diff
            
def plotGraph(minmax, n, diff):
    plt.plot([item[0] for item in minmax],[item[3] for item in minmax])
    plt.title(f'Understanding the curse of dimensionality for n={n} ({diff})')
    plt.xlabel('Value of k')
    plt.ylabel('r(k)')
    plt.savefig(f'{n}_{diff}.jpg')
    plt.show()
    return

for _n in n:
    minmax=[]
    for dim in k:
        dp=[] # data points
        noOfRuns=10 # number of runs
        nOfI=[]
        maxval=0
        minval=0
        rk=0.0
        for i in range(0,noOfRuns):
            dp = np.random.random((_n,dim))
            diff = calcDiff_euc(dp, _n)
            diff = np.asarray(diff)
            diff = np.sort(diff)
            maxval=diff[-1]
            minval=diff[0]
            rk=math.log10((maxval-minval)/(np.mean(diff)))
            nOfI.insert(i,rk)
            rk=np.mean(nOfI)
        minmax.insert(dim,[dim,maxval,minval,rk])
    plotGraph(minmax, _n, "Euclidean")


for _n in n:
    minmax=[]
    for dim in k:
        dp=[] # data points
        noOfRuns=10 # number of runs
        nOfI=[]
        maxval=0
        minval=0
        rk=0.0
        for i in range(0,noOfRuns):
            dp = np.random.random((_n,dim))
            diff = calcDiff_cos(dp, _n)
            diff = np.asarray(diff)
            diff = np.sort(diff)
            #structure of min max
            maxval=diff[-1]
            minval=diff[0]
            rk=math.log10((maxval-minval)/np.mean(diff))
            nOfI.insert(i,rk)
            rk=np.mean(nOfI)
        minmax.insert(dim,[dim,maxval,minval,rk])
    plotGraph(minmax, _n, "Cosine")