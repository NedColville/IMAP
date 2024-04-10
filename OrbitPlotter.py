# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:32:28 2023

@author: Ned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DSCOVR=np.array(pd.read_csv("DSCOVRYZ.csv"))[:int(8614/2)].T
#ACE=np.array(pd.read_csv("ACEYZ.csv"))[:int(8772/2)].T
#WIND=np.array(pd.read_csv("WINDYZ.csv"))[:int(8772/2)].T
#IMAP=pd.read_csv("IMAPOrb.csv",skiprows=2)[:8748]
DSCOVR=np.array(pd.read_csv("DSCOVRYZ.csv")).T
ACE=np.array(pd.read_csv("ACEYZ.csv")).T
WIND=np.array(pd.read_csv("WINDYZ.csv")).T
IMAP=np.array(pd.read_csv("IMAPOrb.csv",skiprows=2)).T


def plot2D(X1,X2,leg,arrows=False,color=0,num_arrows=3):
    X1=list(X1)
    X2=list(X2)

    if arrows==True:
        counter=0
        spacing=int(len(X1)/num_arrows)
        for i in range(0,len(X1)):
            if counter % spacing ==0 and counter !=0:
                grad=np.array([X1[i+1]-X1[i],X2[i+1]-X2[i]])
                plt.quiver(X1[i],X2[i],grad[0],grad[1],color=color, width=0.01)
            counter+=1
        
    plt.plot(X1,X2,label=leg,color=color, lw=0.8)
    return
re=6378
plot2D(np.array(ACE[2])/re,np.array(ACE[3])/re,"ACE",True,'blue')
plot2D(np.array(DSCOVR[2])/re,np.array(DSCOVR[3])/re,"DSCOVR",True,'red')
plot2D(np.array(WIND[2])/re,np.array(WIND[3])/re,"WIND",True,'green')
#plot2D(np.array(IMAP[2])/re,np.array(IMAP[3])/re, "IMAP", True, 'orange')
plt.xlabel("$y_{GSE}, R_E$", fontsize=13)
plt.ylabel("$z_{GSE}, R_E$",fontsize=13)
plt.axis('scaled')
plt.tight_layout()

def circle(r,length,centre):
    theta=np.linspace(0,2*np.pi,length)
    x=centre[0]+r*np.cos(theta)
    y=centre[1]+r*np.sin(theta)
    return [x,y]
test=pd.read_csv("DSCOVRSpeedClean.csv")[72:]
vx=np.array(test["Vx Velocity,km/s"])
x=np.array(6378.1*test["Wind, Xgse,Re"])
time=np.mean(x/vx)
circ=circle(20,1000,[time*30/6378,0])
plt.plot(*circ,label="Magnetosphere",linestyle='dashed',color='black')
dummy_plot = plt.plot([], [], 'o', markersize=3, color='blue', label='Earth')
circle1 = plt.Circle((time*30/6378, 0), 1, color='Blue')
plt.gca().add_patch(circle1)

plt.ylim(-55,30)
plt.tight_layout()
#plt.title("Two Years of Spacecraft Orbit")
plt.legend(ncol=3,loc='lower center')
plt.grid(alpha=0.5)
plt.show()
#%%
from datetime import datetime, timedelta
data=[ACE,DSCOVR,WIND,IMAP]
nameys=['ACE',"DSCOVR","WIND","IMAP"]
offs=[]
re=6378
mins=[]
dates=[]
maxes=[]
newDat=[]
for i in range(len(data)):
    toffs=(np.sqrt((data[i][2].astype(float)-30*60*60)**2+(data[i][3].astype(float))**2))
    if i==3:
        date_objects=[datetime.strptime(date_string, '%Y %b %d %H:%M:%S.%f') for date_string in data[i][0]]
        date_offset = dates[0][0]-date_objects[3000]
        shifted_dates = np.array(date_objects) + date_offset
        dates.append(shifted_dates)
        mins.append(shifted_dates[0])
        maxes.append(shifted_dates[-1])
        #plt.plot(shifted_dates,toffs/re)
    else:
        date_objects = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in data[i][0]]
        dates.append(np.array(date_objects))
        mins.append(date_objects[0])
        maxes.append(date_objects[-1])
        #plt.plot(date_objects,toffs/re)
    newDat.append([dates[-1],toffs])
        
    offs.append(toffs)

"""   
common= [max(mins),min(maxes)] 
masked=[]
for i in range(len(offs)):
    mask = (dates[i] >= common[0]) & (dates[i] <= common[1])
    plt.plot(np.array(dates[i])[mask],np.array(offs[i])[mask]/re)
    masked.append([np.array(dates[i])[mask],np.array(offs[i])[mask]/re])
"""

from alignInterpNew import alignInterp
newDat=alignInterp(newDat,17000)
for i in range(len(newDat[1])):
    plt.plot(newDat[0],np.array(newDat[1][i][0])/re)


plt.xticks(rotation=45)
plt.ylabel("Transverse Offset from Earth, $R_E$")
plt.axhline(45,linestyle='dashed',color='black')
plt.axhline(70,linestyle='dashed',color='red')
plt.axhline(100,linestyle='dashed',color='purple')
plt.legend(['ACE','DSCOVR','WIND','IMAP','$B_{x,y,z}$','$v_{x,y,z}$',r'$\rho$'])
plt.show()




threshold = 45*re

# Initialize a list to store the results for each trajectory
results = []

# Iterate over each timestep
for i in range(len(newDat[1])):
    # Initialize a list to store the results for this timestep
    timestep_results = []
    
    # Check each trajectory's coordinate at this timestep against the threshold
    for trajectory in newDat[1][i][0]:
        if trajectory < threshold:
            timestep_results.append(1)
        else:
            timestep_results.append(0)
    
    # Append the results for this timestep to the overall results list
    results.append(timestep_results)
    

plt.hist([*results])
plt.legend(nameys)
resultsT=np.array(results).T
sums=np.sum(resultsT,axis=1)
plt.show()



unique_values, counts = np.unique(sums, return_counts=True)

# Plot bar chart
percentages=100*counts/np.sum(counts)
plt.grid(alpha=0.5)
plt.bar(unique_values, percentages, alpha=1)
plt.xlabel('Values') 
plt.ylabel('Percentages')
plt.title('Number of Spacecraft Within $B_z$ Correlation Length')
plt.xticks(unique_values)  # Set xticks to unique values
plt.show()
#%%
from datetime import datetime, timedelta
data=[ACE,DSCOVR,WIND]
nameys=['ACE',"DSCOVR","WIND"]
offs=[]
re=6378
mins=[]
dates=[]
maxes=[]
newDat=[]
for i in range(len(data)):
    toffs=(np.sqrt((data[i][2].astype(float)-30*60*60)**2+(data[i][3].astype(float))**2))
    if i==3:
        date_objects=[datetime.strptime(date_string, '%Y %b %d %H:%M:%S.%f') for date_string in data[i][0]]
        date_offset = dates[0][0]-date_objects[3000]
        shifted_dates = np.array(date_objects) + date_offset
        dates.append(shifted_dates)
        mins.append(shifted_dates[0])
        maxes.append(shifted_dates[-1])
        #plt.plot(shifted_dates,toffs/re)
    else:
        date_objects = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in data[i][0]]
        dates.append(np.array(date_objects))
        mins.append(date_objects[0])
        maxes.append(date_objects[-1])
        #plt.plot(date_objects,toffs/re)
    newDat.append([dates[-1],toffs])
        
    offs.append(toffs)

"""   
common= [max(mins),min(maxes)] 
masked=[]
for i in range(len(offs)):
    mask = (dates[i] >= common[0]) & (dates[i] <= common[1])
    plt.plot(np.array(dates[i])[mask],np.array(offs[i])[mask]/re)
    masked.append([np.array(dates[i])[mask],np.array(offs[i])[mask]/re])
"""

from alignInterpNew import alignInterp
newDat=alignInterp(newDat,17000)
for i in range(len(newDat[1])):
    plt.plot(newDat[0],np.array(newDat[1][i][0])/re)


plt.xticks(rotation=45)
plt.ylabel("Transverse Offset from Earth, $R_E$")
plt.axhline(45,linestyle='dashed',color='black')
plt.axhline(70,linestyle='dashed',color='red')
plt.axhline(100,linestyle='dashed',color='purple')
plt.grid(alpha=0.5)
plt.legend(['ACE','DSCOVR','WIND','$B_{x,y,z}$','$v_{x,y,z}$',r'$\rho$'])
plt.show()




threshold = 45*re

# Initialize a list to store the results for each trajectory
results = []

# Iterate over each timestep
for i in range(len(newDat[1])):
    # Initialize a list to store the results for this timestep
    timestep_results = []
    
    # Check each trajectory's coordinate at this timestep against the threshold
    for trajectory in newDat[1][i][0]:
        if trajectory < threshold:
            timestep_results.append(1)
        else:
            timestep_results.append(0)
    
    # Append the results for this timestep to the overall results list
    results.append(timestep_results)
    

plt.hist([*results])
plt.legend(nameys)
resultsT=np.array(results).T
sums=np.sum(resultsT,axis=1)
plt.show()



unique_values1, counts1 = np.unique(sums, return_counts=True)
bwidth=0.35
# Plot bar chart
percentages1=100*counts1/np.sum(counts1)
plt.bar(unique_values1-bwidth/2, percentages1, alpha=1,width=bwidth)
plt.bar(unique_values+bwidth/2, percentages, alpha=1,width=bwidth)

plt.xlabel('Number of Spacecraft Within $B_z$ Correlation Length') 
plt.ylabel('% of the Time')
plt.legend(["Without IMAP", "With IMAP"])
#plt.grid(alpha=0.5)
plt.xticks(np.arange(0,5,1)) # Set xticks to unique values
plt.show()



#%%
#This is very lazy and just copypasting the code but - if we say ACE is shit and use just DSC and Wind instead...
from datetime import datetime, timedelta
data=[DSCOVR,WIND]
nameys=["DSCOVR","WIND"]
offs=[]
re=6378
mins=[]
dates=[]
maxes=[]
newDat=[]
for i in range(len(data)):
    toffs=(np.sqrt((data[i][2].astype(float)-30*60*60)**2+(data[i][3].astype(float))**2))
    if i==2:
        date_objects=[datetime.strptime(date_string, '%Y %b %d %H:%M:%S.%f') for date_string in data[i][0]]
        date_offset = dates[0][0]-date_objects[3000]
        shifted_dates = np.array(date_objects) + date_offset
        dates.append(shifted_dates)
        mins.append(shifted_dates[0])
        maxes.append(shifted_dates[-1])
        #plt.plot(shifted_dates,toffs/re)
    else:
        date_objects = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in data[i][0]]
        dates.append(np.array(date_objects))
        mins.append(date_objects[0])
        maxes.append(date_objects[-1])
        #plt.plot(date_objects,toffs/re)
    newDat.append([dates[-1],toffs])
        
    offs.append(toffs)

"""   
common= [max(mins),min(maxes)] 
masked=[]
for i in range(len(offs)):
    mask = (dates[i] >= common[0]) & (dates[i] <= common[1])
    plt.plot(np.array(dates[i])[mask],np.array(offs[i])[mask]/re)
    masked.append([np.array(dates[i])[mask],np.array(offs[i])[mask]/re])
"""

from alignInterpNew import alignInterp
newDat=alignInterp(newDat,17000)
for i in range(len(newDat[1])):
    plt.plot(newDat[0],np.array(newDat[1][i][0])/re)


plt.xticks(rotation=45)
plt.ylabel("Transverse Offset from Earth, $R_E$")
plt.axhline(45,linestyle='dashed',color='black')
plt.axhline(70,linestyle='dashed',color='red')
plt.axhline(100,linestyle='dashed',color='purple')
plt.legend(['DSCOVR','WIND','$B_{x,y,z}$','$v_{x,y,z}$',r'$\rho$'])
plt.show()




threshold = 45*re

# Initialize a list to store the results for each trajectory
results = []

# Iterate over each timestep
for i in range(len(newDat[1])):
    # Initialize a list to store the results for this timestep
    timestep_results = []
    
    # Check each trajectory's coordinate at this timestep against the threshold
    for trajectory in newDat[1][i][0]:
        if trajectory < threshold:
            timestep_results.append(1)
        else:
            timestep_results.append(0)
    
    # Append the results for this timestep to the overall results list
    results.append(timestep_results)
plt.hist([*results])
plt.legend(nameys)
resultsT=np.array(results).T
sums=np.sum(resultsT,axis=1)
plt.show()



unique_values, counts = np.unique(sums, return_counts=True)

# Plot bar chart
percentages=100*counts/np.sum(counts)
plt.bar(unique_values, percentages, alpha=1)
plt.grid(alpha=0.5)
plt.xlabel('Number of Spacecraft Within $B_z$ Correlation Length') 
plt.ylabel('Proportion of the Time')
#plt.title('Number of Spacecraft Within $B_z$ Correlation Length')
plt.xticks(unique_values)  # Set xticks to unique values
plt.show()

#%%
#This is EVEN LAZIER and just repeating the code but - if we say ACE is shit and use IMAP instead...
from datetime import datetime, timedelta
data=[DSCOVR,WIND,IMAP]
nameys=["DSCOVR","WIND","IMAP"]
offs=[]
re=6378
mins=[]
dates=[]
maxes=[]
newDat=[]
for i in range(len(data)):
    toffs=(np.sqrt((data[i][2].astype(float)-30*60*60)**2+(data[i][3].astype(float))**2))
    if i==2:
        date_objects=[datetime.strptime(date_string, '%Y %b %d %H:%M:%S.%f') for date_string in data[i][0]]
        date_offset = dates[0][0]-date_objects[3000]
        shifted_dates = np.array(date_objects) + date_offset
        dates.append(shifted_dates)
        mins.append(shifted_dates[0])
        maxes.append(shifted_dates[-1])
        #plt.plot(shifted_dates,toffs/re)
    else:
        date_objects = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in data[i][0]]
        dates.append(np.array(date_objects))
        mins.append(date_objects[0])
        maxes.append(date_objects[-1])
        #plt.plot(date_objects,toffs/re)
    newDat.append([dates[-1],toffs])
        
    offs.append(toffs)

"""   
common= [max(mins),min(maxes)] 
masked=[]
for i in range(len(offs)):
    mask = (dates[i] >= common[0]) & (dates[i] <= common[1])
    plt.plot(np.array(dates[i])[mask],np.array(offs[i])[mask]/re)
    masked.append([np.array(dates[i])[mask],np.array(offs[i])[mask]/re])
"""

from alignInterpNew import alignInterp
newDat=alignInterp(newDat,17000)
for i in range(len(newDat[1])):
    plt.plot(newDat[0],np.array(newDat[1][i][0])/re)


plt.xticks(rotation=45)
plt.ylabel("Transverse Offset from Earth, $R_E$")
plt.axhline(45,linestyle='dashed',color='black')
plt.axhline(70,linestyle='dashed',color='red')
plt.axhline(100,linestyle='dashed',color='purple')
plt.legend(['DSCOVR','WIND','IMAP','$B_{x,y,z}$','$v_{x,y,z}$',r'$\rho$'])
plt.show()




threshold = 45*re

# Initialize a list to store the results for each trajectory
results = []

# Iterate over each timestep
for i in range(len(newDat[1])):
    # Initialize a list to store the results for this timestep
    timestep_results = []
    
    # Check each trajectory's coordinate at this timestep against the threshold
    for trajectory in newDat[1][i][0]:
        if trajectory < threshold:
            timestep_results.append(1)
        else:
            timestep_results.append(0)
    
    # Append the results for this timestep to the overall results list
    results.append(timestep_results)
plt.hist([*results])
plt.legend(nameys)
resultsT=np.array(results).T
sums=np.sum(resultsT,axis=1)
plt.show()



unique_values2, counts2 = np.unique(sums, return_counts=True)

# Plot bar chart
bwidth=0.35
plt.grid(alpha=0.5)
percentages2=100*counts2/np.sum(counts2)
plt.bar(unique_values-bwidth/2, percentages, width=bwidth,alpha=1,label="Without IMAP")
plt.bar(unique_values2+bwidth/2, percentages2, width=bwidth,alpha=1,label="With IMAP")
plt.xlabel('Number of Spacecraft Within $B_z$ Correlation Length')
plt.ylabel('% of the Time')
plt.legend()
#plt.title('Number of Spacecraft Within $B_z$ Correlation Length')
plt.xticks(np.arange(0,4,1))  # Set xticks to unique values
plt.show()







    









