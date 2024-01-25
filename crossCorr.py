# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:21:41 2023

@author: Ned
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from timefuncs import datetime_to_unix_seconds, unix_seconds_to_datetime
def MaxCrossCorr(t,x,y,legend = ["X", "Y"]):
    plt.plot(t,x)
    plt.plot(t,y)
    plt.xlabel("Time")
    plt.ylabel("$B_z$ nT")
    plt.legend(legend)
    plt.show()
    vals=[]
    vals=np.correlate(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    shiftedX=np.roll(x,np.argmax(vals))
    plt.plot(t,shiftedX)
    plt.plot(t,y)
    plt.xlabel("Time")
    plt.ylabel("$B_z$ nT")
    legend[0]+= " Shifted"
    plt.legend(legend)
    plt.show()
    plt.plot(np.array(t),vals)
    plt.xlabel("Time After First Measurement, s")
    plt.ylabel("Correlation Coefficient")
    plt.show()
    return vals, t[np.argmax(vals)]

def find_closest_index(lst, target_value):
    """
    Find the index of the value in the list closest to the target value.

    Parameters:
    - lst: The list of values.
    - target_value: The value to which the closest value in the list will be found.

    Returns:
    - The index of the closest value in the list.
    """
    closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - target_value))
    return closest_index

def crossCorrShift(t, x, y, tshift, plot=False):
    """
    Computes the cross-correlation between two signals and determines the time shift based on a provided prediction.
    
    Parameters:
    t (array-like): Time axis for the signals.
    x (array-like): First signal array.
    y (array-like): Second signal array.
    tshift (float): Predicted time shift for aligning signals.
    plot (bool, optional): Controls whether to plot signals and correlation results. Defaults to False.
    
    Returns:
    tuple: A tuple containing:
        - vals (array-like): Array of correlation coefficients for various time shifts.
        - offSet (float): Time offset between signals based on the prediction.
        - zeroCorr (float): Correlation coefficient at zero time shift.
    """

    tTemp = t-t[0]
    if plot:
        tplot=unix_seconds_to_datetime(t)
        plt.plot(tplot,x)
        plt.plot(tplot,y)
        plt.title("No shift")
        plt.show()
    indShift = int(find_closest_index(tTemp, tshift))
    y = np.roll(y, indShift)
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    if plot:
        plt.plot(tplot, x)
        plt.plot(tplot, y)
        plt.title("Calculated shift")
        plt.show()
    vals = np.correlate(x, y, 'same') / (norm)

    #plt.plot(vals)
    #plt.show()
    if plot:
        tTemp = tTemp - np.mean(tTemp)
        plt.plot(tTemp, vals)
        plt.axvline(0, linestyle='dashed')
        plt.xlabel("$\Delta t$")
        plt.ylabel("Correlation Coefficient")
        plt.show()
    else:
        tTemp = tTemp - np.mean(tTemp)
    # Uncommented lines for testing
    #vals = vals[::-1]
    offSet = tTemp[np.argmax(vals)] - tTemp[find_closest_index(tTemp, 0)]
    vals = np.roll(vals, int(len(vals)/2))   

    zeroCorr=vals[0]

    return vals, offSet, zeroCorr, np.max(vals)



    

