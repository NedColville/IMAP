# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:45:04 2024

@author: Ned
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
times=np.load("split_data_times_ALLf_corrected.npy")
ACE=pd.read_csv("ace_data_unix.csv")
Wind=pd.read_csv("wind_data_unix.csv")
DSCOVR=pd.read_csv("dscovr_data_unix.csv")
SYM=pd.read_csv("SYM_data_unix.csv")

Re=6378
DSCOVR['Field Magnitude,nT'] = DSCOVR['Field Magnitude,nT'].replace(9999.99, np.nan)
DSCOVR['Vector Mag.,nT'] = DSCOVR['Vector Mag.,nT'].replace(9999.99, np.nan)
DSCOVR['BX, GSE, nT'] = DSCOVR['BX, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['BY, GSE, nT'] = DSCOVR['BY, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['BZ, GSE, nT'] = DSCOVR['BZ, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['Speed, km/s'] = DSCOVR['Speed, km/s'].replace(99999.9, np.nan)
DSCOVR['Vx Velocity,km/s'] = DSCOVR['Vx Velocity,km/s'].replace(99999.9, np.nan)
DSCOVR['Vy Velocity, km/s'] = DSCOVR['Vy Velocity, km/s'].replace(99999.9, np.nan)
DSCOVR['Vz Velocity, km/s'] = DSCOVR['Vz Velocity, km/s'].replace(99999.9, np.nan)
DSCOVR['Proton Density, n/cc'] = DSCOVR['Proton Density, n/cc'].replace(999.999, np.nan)
DSCOVR['Wind, Xgse,Re'] = DSCOVR['Wind, Xgse,Re'].replace(9999.99, np.nan)
DSCOVR['Wind, Ygse,Re'] = DSCOVR['Wind, Ygse,Re'].replace(9999.99, np.nan)
DSCOVR['Wind, Zgse,Re'] = DSCOVR['Wind, Zgse,Re'].replace(9999.99, np.nan)

# Have to do change to km AFTER removing vals else fill value will change!
DSCOVR['Wind, Xgse,Re'] = DSCOVR['Wind, Xgse,Re']*Re
DSCOVR['Wind, Ygse,Re'] = DSCOVR['Wind, Ygse,Re']*Re
DSCOVR['Wind, Zgse,Re'] = DSCOVR['Wind, Zgse,Re']*Re

# Interpolate NaN values
for column in DSCOVR.columns:
    DSCOVR[column] = DSCOVR[column].interpolate()
    
# Define the desired order of columns as required for Ned's function: DSCOVR
desired_columns_order = [
    'Time',
    'Vx Velocity,km/s',
    'Vy Velocity, km/s',
    'Vz Velocity, km/s',
    'Wind, Xgse,Re',
    'Wind, Ygse,Re',
    'Wind, Zgse,Re',
    'BZ, GSE, nT',
    'Speed, km/s',
    'Proton Density, n/cc'
]

# Select only the desired columns and reorder them
DSCOVR = DSCOVR[desired_columns_order]

DSCOVR = DSCOVR.copy()

# Drop the original columns
DSCOVR = DSCOVR.drop(['Proton Density, n/cc', 'Speed, km/s', 'BZ, GSE, nT'], axis=1)




Wind['BX, GSE, nT'] = Wind['BX, GSE, nT'].replace(9999.990000, np.nan)
Wind['BY, GSE, nT'] = Wind['BY, GSE, nT'].replace(9999.990000, np.nan)
Wind['BZ, GSE, nT'] = Wind['BZ, GSE, nT'].replace(9999.990000, np.nan)
Wind['Vector Mag.,nT'] = Wind['Vector Mag.,nT'].replace(9999.990000, np.nan)
Wind['Field Magnitude,nT'] = Wind['Field Magnitude,nT'].replace(9999.990000, np.nan)
Wind['KP_Vx,km/s'] = Wind['KP_Vx,km/s'].replace(99999.900000, np.nan)
Wind['Kp_Vy, km/s'] = Wind['Kp_Vy, km/s'].replace(99999.900000, np.nan)
Wind['KP_Vz, km/s'] = Wind['KP_Vz, km/s'].replace(99999.900000, np.nan)
Wind['KP_Speed, km/s'] = Wind['KP_Speed, km/s'].replace(99999.900000, np.nan)
Wind['Kp_proton Density, n/cc'] = Wind['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
Wind['Wind, Xgse,Re'] = Wind['Wind, Xgse,Re'].replace(9999.990000, np.nan)
Wind['Wind, Ygse,Re'] = Wind['Wind, Ygse,Re'].replace(9999.990000, np.nan)
Wind['Wind, Zgse,Re'] = Wind['Wind, Zgse,Re'].replace(9999.990000, np.nan)

Wind['Wind, Xgse,Re'] = Wind['Wind, Xgse,Re']*Re
Wind['Wind, Ygse,Re'] = Wind['Wind, Ygse,Re']*Re
Wind['Wind, Zgse,Re'] = Wind['Wind, Zgse,Re']*Re


# Interpolate NaN values
for column in Wind.columns:
    Wind[column] = Wind[column].interpolate()
    
# Define the desired order of columns as required for Ned's function: WIND
desired_columns_order = [
    'Time',
    'KP_Vx,km/s',
    'Kp_Vy, km/s',
    'KP_Vz, km/s',
    'Wind, Xgse,Re',
    'Wind, Ygse,Re',
    'Wind, Zgse,Re',
    'BZ, GSE, nT',
    'KP_Speed, km/s',
    'Kp_proton Density, n/cc'
]

# Select only the desired columns and reorder them
Wind = Wind[desired_columns_order]


# Drop the original columns
Wind = Wind.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1)

for column in ACE.columns:
    ACE[column] = ACE[column].interpolate()
    
# Define the desired order of columns as required for Ned's function: ACE
desired_columns_order = [
    'Time',
    'vx',
    'vy',
    'vz',
    'x',
    'y',
    'z',
    'Bz',
    'n'
]

# Select only the desired columns and reorder them
ACE = ACE[desired_columns_order]

# Need a velocity magnitude column for ACE

# Drop the original columns
ACE.drop(['Bz', 'n'], axis=1, inplace=True)




DSCOVR.rename(columns={'Vx Velocity,km/s': 'vx',
                       'Vy Velocity, km/s': 'vy',
                       'Vz Velocity, km/s':'vz',
                       'Wind, Xgse,Re': 'x',
                       'Wind, Ygse,Re': 'y',
                       'Wind, Zgse,Re': 'z'}, inplace=True)
Wind.rename(columns={'KP_Vx,km/s': 'vx',
                     'Kp_Vy, km/s': 'vy',
                     'KP_Vz, km/s': 'vz',
                     'Wind, Xgse,Re': 'x',
                     'Wind, Ygse,Re': 'y',
                     'Wind, Zgse,Re': 'z'}, inplace=True)



#%%
dataset=[ACE,Wind,DSCOVR]

bigYs=[]
bigZs=[]
for i in range(len(times)):
    print(i)    
    for data in dataset:
        dfFilt = data[(SYM['Time'] >= times[i][0]) & (SYM['Time'] <= times[i][1])]
        ts=-1*np.array(dfFilt['x']-10*Re)/np.array(dfFilt['vx'])
        ynew=np.array((dfFilt['y'])+(np.array(dfFilt['vy']))*ts)/Re
        znew=np.array((dfFilt['z'])+(np.array(dfFilt['vz']))*ts)/Re
        bigYs.append(ynew)
        bigZs.append(znew)

bigYs=np.array(bigYs).ravel()
bigZs=np.array(bigZs).ravel()
#%%
from matplotlib.colors import LogNorm
bigYsN = bigYs[(abs(bigYs) < 130) & (abs(bigZs) < 60)]
bigZsN = bigZs[(abs(bigYs) < 130) & (abs(bigZs) < 60)]
plt.hist2d(bigYsN,bigZsN,bins=150,cmap='viridis')
plt.colorbar(label='Frequency',shrink=0.7)

# Add labels and title
plt.xlabel('$y_{GSE}, R_E$')
plt.ylabel('$z_{GSE}, R_E$')
#plt.title('2D Histogram with Colorbar')
plt.gca().set_aspect('equal')

# Show plot
def contour_function(x, y):
    return np.exp((-np.sqrt((x+(3600*30)/6378)**2 + y**2))/45)

# Generate contours from the contour function
x_contour = np.linspace(-130, 130, 100)
y_contour = np.linspace(-60, 60, 100)
X_contour, Y_contour = np.meshgrid(x_contour, y_contour)
Z_contour = contour_function(X_contour, Y_contour)



# Plot contours
contour_lines = plt.contour(X_contour, Y_contour, Z_contour, colors='orange',alpha=0.8,levels=5)

# Add contour labels
plt.clabel(contour_lines, inline=True, fontsize=8)
plt.show()


    