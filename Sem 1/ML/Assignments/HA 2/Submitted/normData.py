import numpy as np
import pandas as pd

def minMaxNorm(origMat):

    noRow = origMat.shape[0]
    noCol = origMat.shape[1]
    finalMat = np.zeros(origMat.shape)

    for i in range(0, noCol):
        minVal = np.min(origMat[:, i])
        maxVal = np.max(origMat[:, i])
        rangeVal = maxVal - minVal
        for j in range(0, noRow):
            newTemp = (origMat[j, i] - minVal) / rangeVal
            finalMat[j, i] = newTemp
    return finalMat


def meanNorm(origMat):

    noRow = origMat.shape[0]
    noCol = origMat.shape[1]
    finalMat = np.zeros(origMat.shape)

    finalMat[:, noCol-1] = origMat[:, noCol-1]

    for i in range(0, noCol-1):
        minVal = np.min(origMat[:, i])
        maxVal = np.max(origMat[:, i])
        rangeVal = maxVal - minVal
        mean = np.mean(origMat[:, i])
        # print(minVal, maxVal, rangeVal, "Min, Max, Ran")
        for j in range(0, noRow):
            newTemp = (origMat[j, i] - mean) / rangeVal
            finalMat[j, i] = newTemp
            # print(origMat)
    return finalMat
