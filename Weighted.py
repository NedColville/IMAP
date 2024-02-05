# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:02:04 2024

@author: Ned
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

"""
DSCOVR=pd.read_csv("testDSCOVR.csv").to_numpy().T
ACE=pd.read_csv("testACE.csv").to_numpy().T
WIND=pd.read_csv("testWIND.csv").to_numpy().T
sList=[DSCOVR,ACE,WIND]
"""

def getWeights(ri):
    r0=50*6378
    return np.exp(-1*ri/r0)

def alignInterp(datasets):
    """
    Align time series within datasets and interpolate data for each dataset at one-minute intervals.

    Parameters:
    datasets: Variable number of datasets, each containing time series and data arrays.

    Returns:
    list of numpy.ndarray: List of datasets with new data and time series.
    """
    # Extract time series and data arrays for each dataset
    time_series_list = [dataset[0] for dataset in datasets]
    data_arrays_list = [dataset[1:] for dataset in datasets]

    # Find the common start and end times for all datasets
    common_start_time = max([min(time) for time in time_series_list])
    common_end_time = min([max(time) for time in time_series_list])

    # Create the common time series within the overlapping time range at one-minute intervals
    interpolated_time_series = np.arange(common_start_time, common_end_time, 60)

    # Interpolate data for each dataset
    interpolated_data = [
        [np.interp(interpolated_time_series, time_series, data) for data in data_arrays]
        for time_series, data_arrays in zip(time_series_list, data_arrays_list)
    ]

    # Create a list of datasets with new data and time series
    result_datasets = [
        [np.array(interpolated_time_series)] + [np.array(data) for data in data_arrays]
        for interpolated_time_series, data_arrays in zip(interpolated_time_series, interpolated_data)
    ]
    for dataset in result_datasets:
        dataset[0]=interpolated_time_series
    return result_datasets



def weightedAv(sList,sRef,indS):
    """
    USAGE: Would recommend calling function and iterating through indS values you want
    No functionality to pass in and iterate through a list of indices.
    Interpolation/propagation/weight determination are all vectorized so should be quick enough

    Parameters
    ----------
    sList : list/array
    List of spacecraft measurements. Elements in this list should be of the form:
    [t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...].T lol.
        
    sRef: list/array                      
    Target: [[t0,x0,y0,z0,DATA],[t1,...].T lol].
    
    indS: int
    Index of data you want to average in s
    
    Returns
    -------
    sListNew: List
    New list of propagated/modified s/c data
    
    [sListNew[0][0],weightedBs]: List
    List containing averaged data and averaged data along with its
    interpolated time series
    
    

    """
    weights=[]
    i=0
    for s in sList:

        Ts=(14*6378-s[4])/(s[1])#calculates prop time
        s=[s[0]+Ts,s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                                   np.array(s[5])+s[2]*Ts,
                                   np.array(s[6])+s[3]*Ts,
                                   s[7],s[8],s[9]] #modifies position and time in list
        sList[i]=s
        i+=1
    sListNew=alignInterp(sList)     
    i=0
    for s in sListNew:
        offsets=np.sqrt(s[5]**2+s[6]**2) #gets offsets and weights
        weights.append(np.array([getWeights(ri) for ri in offsets]))
        sListNew[i]=s
        i+=1
    weights=np.array(weights).T

    weightedBs=[]
    for i in range(0,len(np.array(sListNew[0]).T)):
        tempBz=[]
        for data in sListNew:
            tempBz.append(data[indS][i]) #weighted averages the quantity s[indS]
        B=sum(np.array(tempBz)*weights[i])/sum(weights[i])
        weightedBs.append(B)
    
    return sListNew,[sListNew[0][0],weightedBs]

"""
length=len(DSCOVR.T)  
OMNIPos=[ACE[0],np.full(length,14*6378),np.zeros(length),np.zeros(length)]

for data in sList:
    plt.plot(data[0],data[7])

plt.show()

test=weightedAv(sList, OMNIPos,1)
"""
    