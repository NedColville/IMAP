# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:30:43 2024

@author: Ned
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def maxLead(s1,s2,s3):
    """
    Parameters
    ----------
    s1 : list/array
    Spacecraft data: [[t0,[vx0,vy0,vz0],[x0,y0,z0],[DATA]],[t1,[vx1,vy1,vz1],[x1,y1,z1],...].T lol..
    s2 : list/array
    Spacecraft data: [[t0,[vx0,vy0,vz0],[x0,y0,z0],[DATA]],[t1,[vx1,vy1,vz1],[x1,y1,z1],...].T lol..
    s3 : list/array
    Spacecraft data: [[t0,[vx0,vy0,vz0],[x0,y0,z0],[DATA]],[t1,[vx1,vy1,vz1],[x1,y1,z1],...].T lol..

    Returns
    -------
    vals : list
    index of spacecraft with highest lead time at each time across period

    """
    vals=[]
    for i in range(0,len(s1[0])):
        s1Lead=s1.T[i][4]/s1.T[i][1]
        s2Lead=s2.T[i][4]/s2.T[i][1]
        s3Lead=s3.T[i][4]/s2.T[i][1]
        vals.append(np.argmax([s1Lead,s2Lead,s3Lead]))
    return vals
def minOffs(s1,s2,s3,sRef,plot=False):
    """
    Parameters
    ----------
    s1 : list/array
    Spacecraft data: [[t0,[vx0,vy0,vz0],[x0,y0,z0],[DATA]],[t1,[vx1,vy1,vz1],[x1,y1,z1],...].T lol..
    s2 : list/array
    Spacecraft data: [[t0,[vx0,vy0,vz0],[x0,y0,z0],[DATA]],[t1,[vx1,vy1,vz1],[x1,y1,z1],...].T lol..
    s3 : list/array
    Spacecraft data: [[t0,[vx0,vy0,vz0],[x0,y0,z0],[DATA]],[t1,[vx1,vy1,vz1],[x1,y1,z1],...].T lol..

    sRef: list/array                      
    Target: [[t0,[x0,y0,z0],[DATA]],[t1,...].T lol]
    
    Returns
    -------
    vals : list
    index of spacecraft with lowest offset at each time across period

    """
    sList=[s1,s2,s3]
    vals=[]
    if plot:
        for s in sList:
            plt.plot(s[5],s[6])
        plt.plot(sRef[2],sRef[3])
        plt.legend(["DSCOVR","ACE","WIND","ART"])
    for i in range(0,len(s1[0])):
        tempOffs=[]
        for s in sList:
            deltay=s.T[i][5]-sRef.T[i][2]
            deltaz=s.T[i][6]-sRef.T[i][3]
            tempOffs.append(np.sqrt(deltay**2+deltaz**2))
        vals.append(np.argmin(tempOffs))
    return vals
"""
DSCOVR=pd.read_csv("testDSCOVR.csv").to_numpy().T
ACE=pd.read_csv("testACE.csv").to_numpy().T
WIND=pd.read_csv("testWIND.csv").to_numpy().T
ART=pd.read_csv("testARTEMIS.csv").to_numpy().T    
test=minOffs(DSCOVR,ACE,WIND,ART)
test2=maxLead(DSCOVR,ACE,WIND)
"""


    






        