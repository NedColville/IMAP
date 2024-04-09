# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:52:53 2024

@author: Ned
"""

#CODE FOR LOG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from cross_correlation import cross_correlation
import scipy.stats as stats

#%%



zvCCsM = np.load('ABC_MvsR_zvCCs_4hrs_newWeights.npy')
mCCsM = np.load('ABC_MvsR_maxCCs_4hrs_newWeights.npy')
dTsM = np.load('ABC_MvsR_deltaTs_4hrs_newWeights.npy')
RMSEsM = np.load('ABC_MvsR_RMSEs_4hrs_newWeights.npy')
MAEsM = np.load('ABC_MvsR_MAEs_4hrs_newWeights.npy')
R2sM = np.load('ABC_MvsR_R2s_4hrs_newWeights.npy')


df_real_symsM = pd.read_csv('ABC_MvsR_real_syms_4hrs_newWeights.csv').T
df_sym_time_seriesM = pd.read_csv('ABC_MvsR_sym_time_series_4hrs_newWeights.csv').T
df_sym_forecastsM = pd.read_csv('ABC_MvsR_sym_forecasts_4hrs_newWeights.csv').T



# Wind
df_EPtimeW = pd.read_csv('ABC_WvsR_EPtime_4hrs_ALL3split.csv').T
df_EforecastW = pd.read_csv('ABC_WvsR_Eforecast_4hrs_ALL3split.csv').T
df_PforecastW = pd.read_csv('ABC_WvsR_Pforecast_4hrs_ALL3split.csv').T

zvCCsW = np.load('ABC_WvsR_zvCCs_4hrs_ALL3split.npy')
mCCsW = np.load('ABC_WvsR_maxCCs_4hrs_ALL3split.npy')
dTsW = np.load('ABC_WvsR_deltaTs_4hrs_ALL3split.npy')


df_real_symsW = pd.read_csv('ABC_WvsR_real_syms_4hrs_ALL3split.csv').T
df_sym_time_seriesW = pd.read_csv('ABC_WvsR_sym_time_series_4hrs_ALL3split.csv').T
df_sym_forecastsW = pd.read_csv('ABC_WvsR_sym_forecasts_4hrs_ALL3split.csv').T
SYM0sW = np.load('ABC_WvsR_SYM0s_4hrs_ALL3split.npy')


# DSCOVR
df_EPtimeD = pd.read_csv('ABC_DvsR_EPtime_4hrs_ALL3split.csv').T
df_EforecastD = pd.read_csv('ABC_DvsR_Eforecast_4hrs_ALL3split.csv').T
df_PforecastD = pd.read_csv('ABC_DvsR_Pforecast_4hrs_ALL3split.csv').T

zvCCsD = np.load('ABC_DvsR_zvCCs_4hrs_ALL3split.npy')
mCCsD = np.load('ABC_DvsR_maxCCs_4hrs_ALL3split.npy')
dTsD = np.load('ABC_DvsR_deltaTs_4hrs_ALL3split.npy')

df_real_symsD = pd.read_csv('ABC_DvsR_real_syms_4hrs_ALL3split.csv').T
df_sym_time_seriesD = pd.read_csv('ABC_DvsR_sym_time_series_4hrs_ALL3split.csv').T
df_sym_forecastsD = pd.read_csv('ABC_DvsR_sym_forecasts_4hrs_ALL3split.csv').T
SYM0sD = np.load('ABC_DvsR_SYM0s_4hrs_ALL3split.npy')


# ACE
df_EPtimeA = pd.read_csv('ABC_AvsR_EPtime_4hrs_ALL3split.csv').T
df_EforecastA = pd.read_csv('ABC_AvsR_Eforecast_4hrs_ALL3split.csv').T
df_PforecastA = pd.read_csv('ABC_AvsR_Pforecast_4hrs_ALL3split.csv').T

zvCCsA = np.load('ABC_AvsR_zvCCs_4hrs_ALL3split.npy')
mCCsA = np.load('ABC_AvsR_maxCCs_4hrs_ALL3split.npy')
dTsA = np.load('ABC_AvsR_deltaTs_4hrs_ALL3split.npy')

df_real_symsA = pd.read_csv('ABC_AvsR_real_syms_4hrs_ALL3split.csv').T
df_sym_time_seriesA = pd.read_csv('ABC_AvsR_sym_time_series_4hrs_ALL3split.csv').T
df_sym_forecastsA = pd.read_csv('ABC_AvsR_sym_forecasts_4hrs_ALL3split.csv').T
SYM0sA = np.load('ABC_AvsR_SYM0s_4hrs_ALL3split.npy')

YZsA = np.load('ABC_AvsR_YZs_4hrs_newWeights.npy')
YZsD = np.load('ABC_DvsR_YZs_4hrs_newWeights.npy')
YZsW = np.load('ABC_WvsR_YZs_4hrs_newWeights.npy')


#%%
#Gets TP,FP,TN,FNs
def compare_forecasts(df_forecasts, df_real, threshold):
    TP = ((df_forecasts < threshold) & (df_real < threshold)).sum().sum()
    FP = ((df_forecasts < threshold) & (df_real >= threshold)).sum().sum()
    TN = ((df_forecasts >= threshold) & (df_real >= threshold)).sum().sum()
    FN = ((df_forecasts >= threshold) & (df_real < threshold)).sum().sum()   
    total_elements = np.count_nonzero(~np.isnan(df_forecasts))  # Count non-NaN values
    #total_elements = df_forecasts.size
    #print(below_10_forecast_below_10_real+above_10_forecast_below_10_real)
    TP = (TP / total_elements) * 100
    FP = (FP / total_elements) * 100
    TN = (TN / total_elements) * 100
    FN = (FN / total_elements) * 100
    #print(total_elements)
    sens= TP/(TP+FN)
    spec= TN/(TN+FP)
    
    
    return [[TP, FP],
            [FN, TN]]
#Masking to keep big events
def mask_forecasts(df_forecasts, df_real,cutoff):
    masked_forecasts = df_forecasts[cutoff >= df_real]
    masked_real = df_real[cutoff >= df_real]
    
    return masked_forecasts, masked_real

#Masking to keep small events
def mask_forecasts2(df_forecasts, df_real,cutoff):
    masked_forecasts = df_forecasts[df_real >= cutoff]
    masked_real = df_real[df_real >= cutoff]
    
    return masked_forecasts, masked_real

threshold=-10 #defines what an event is
#These are the values for the basic conf matrix - no masking, event =-10
percentagesM = compare_forecasts(df_sym_forecastsM, df_real_symsM,threshold)
percentagesA = compare_forecasts(df_sym_forecastsA, df_real_symsA,threshold)
percentagesD = compare_forecasts(df_sym_forecastsD, df_real_symsD,threshold)
percentagesW = compare_forecasts(df_sym_forecastsW, df_real_symsW,threshold)
def getSens(conf):
    sens=(conf[0][0])/(conf[0][0]+conf[1][0])
    return sens
def getSpec(conf):
    if conf[1][1]+ conf[0][1]==0:
        return np.nan
    else:
        
        spec=(conf[1][1])/(conf[1][1]+conf[0][1])
        return spec

#%%
#nice conf matrix plotter
def makeMatrix(conf_matrix,label):
    conf_matrix_labels = np.array([['{:.2f}%'.format(conf_matrix[0][0]), '{:.2f}%'.format(conf_matrix[0][1])],
                                   ['{:.2f}%'.format(conf_matrix[1][0]), '{:.2f}%'.format(conf_matrix[1][1])]])


    # Plotting
    labels = ['Predict an Event', 'False Positive', 'False Negative', 'True Positive']
    categories = ['Negative', 'Positive']
    sns.heatmap(conf_matrix, annot=conf_matrix_labels, fmt="", cmap='Blues', 
                xticklabels=['Event Occurs', 'No Event'], 
                yticklabels=['Predict an Event', 'Predict No Event'], annot_kws={"size": 14})
    plt.xlabel('Measurements', fontsize=16)
    plt.ylabel('Predictions', fontsize=16)
    plt.title(label+' Confusion Matrix, Event = '+str(threshold)+" nT", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sens=getSens(conf_matrix)
    spec=getSpec(conf_matrix)
    print(label +" Sensitivity = "+str(round(sens,3)))
    print(label +" Specificity = "+str(round(spec,3)))
    print(label +" Mean = "+str(round(sens+spec,4)/2))
    print()
    plt.show()
    return sens, spec
    
    

makeMatrix(percentagesD,'DSCOVR')
makeMatrix(percentagesW,'Wind')
makeMatrix(percentagesA,'ACE')
makeMatrix(percentagesM,'Multi')

#%%
#Plotting sensitivity for different event sizes
#Important information is at the start of our range - so log spacing might be useful

threshes=np.linspace(-10,-100,21)
logThresh=np.logspace(1,1.65,10)*-1
statsM=[]
statsD=[]
statsA=[]
statsW=[]
for threshold in logThresh: #change with logThresh or threshes here - remember to change what you're plotting accordingly
    temp=compare_forecasts(*mask_forecasts(df_sym_forecastsM, df_real_symsM,threshold),-10)
    statsM.append([getSens(temp),getSpec(temp)])
    temp=compare_forecasts(*mask_forecasts(df_sym_forecastsW, df_real_symsW,threshold),-10)
    statsW.append([getSens(temp),getSpec(temp)])
    temp=compare_forecasts(*mask_forecasts(df_sym_forecastsD, df_real_symsD,threshold),-10)
    statsD.append([getSens(temp),getSpec(temp)])
    temp=compare_forecasts(*mask_forecasts(df_sym_forecastsA, df_real_symsA,threshold),-10)
    statsA.append([getSens(temp),getSpec(temp)])
statsM=np.array(statsM).T
statsW=np.array(statsW).T
statsD=np.array(statsD).T
statsA=np.array(statsA).T

#Spacing the points out - looks kind of cool when points are v. close

# plt.plot(logThresh+0.25,statsD[0],'x')
# plt.plot(logThresh-0.25,statsW[0],'x')
# plt.plot(logThresh-0.75,statsA[0],'x')
# plt.plot(logThresh+0.75,statsM[0],'x') 

plt.plot(logThresh,statsD[0],'x')
plt.plot(logThresh,statsW[0],'x')
plt.plot(logThresh,statsA[0],'x')
plt.plot(logThresh,statsM[0],'x')
meths=["ACE","DSCOVR","Wind","Multi"]
plt.legend(meths)
plt.grid(alpha=0.5)
#plt.xscale('log')
plt.xlabel("SYM-H Masking Threshold")
plt.ylabel("Sensitivity")
plt.title("Sensitivity for Different Event Sizes (More eventful than $x$ )")
plt.show()
#plt.plot(logThresh,statsM[0]-statsA[0])

#%%
#Plotting specificity for different event sizes
threshes=np.linspace(-10,15,21)
#logThresh=np.logspace(1,2,21)*-1
statsM=[]
statsD=[]
statsA=[]
statsW=[]
for threshold in threshes:
    temp=compare_forecasts(*mask_forecasts2(df_sym_forecastsM, df_real_symsM,threshold),-10)
    statsM.append([getSens(temp),getSpec(temp)])
    temp=compare_forecasts(*mask_forecasts2(df_sym_forecastsW, df_real_symsW,threshold),-10)
    statsW.append([getSens(temp),getSpec(temp)])
    temp=compare_forecasts(*mask_forecasts2(df_sym_forecastsD, df_real_symsD,threshold),-10)
    statsD.append([getSens(temp),getSpec(temp)])
    temp=compare_forecasts(*mask_forecasts2(df_sym_forecastsA, df_real_symsA,threshold),-10)
    statsA.append([getSens(temp),getSpec(temp)])
statsM=np.array(statsM).T
statsW=np.array(statsW).T
statsD=np.array(statsD).T
statsA=np.array(statsA).T
        
plt.plot(threshes,statsM[1],'x')
plt.plot(threshes,statsD[1],'x')
plt.plot(threshes,statsW[1],'x')
plt.plot(threshes,statsA[1],'x')
meths=["ACE","DSCOVR","Wind","Multi"]
plt.legend(meths)  
plt.xlabel("SYM-H Masking Threshold")
plt.ylabel("Specificity")
plt.title("Specificity for Different Event Sizes (Less eventful than $x$ )")
plt.grid(alpha=0.5)

#%%
#Bootstrapping to get variation on FN,FP,TP,TN


from scipy.stats import norm

def compare_forecastsBoot(df_forecasts, df_real, threshold,conf, n_bootstraps=1000):
    # Initialize lists to store bootstrapped results
    bootstrapped_TPs = []
    bootstrapped_FPs = []
    bootstrapped_TNs = []
    bootstrapped_FNs = []

    # Iterate through bootstraps
    for _ in range(n_bootstraps):
        if _%100==0:
            print(_)
        # Sample with replacement from the indices of the dataframes
        sampled_indices = np.random.choice(df_forecasts.index, size=len(df_forecasts), replace=True)
        
        # Sample the dataframes using the sampled indices
        sampled_df_forecasts = df_forecasts.loc[sampled_indices]
        sampled_df_real = df_real.loc[sampled_indices]
        
        # Calculate TP, FP, TN, FN for the bootstrapped sample
        TP = ((sampled_df_forecasts < threshold) & (sampled_df_real < threshold)).sum().sum()
        FP = ((sampled_df_forecasts < threshold) & (sampled_df_real >= threshold)).sum().sum()
        TN = ((sampled_df_forecasts >= threshold) & (sampled_df_real >= threshold)).sum().sum()
        FN = ((sampled_df_forecasts >= threshold) & (sampled_df_real < threshold)).sum().sum()

        # Calculate the total number of elements (non-NaN) in the bootstrapped sample
        total_elements = np.count_nonzero(~np.isnan(sampled_df_forecasts))

        # Calculate percentages for TP, FP, TN, FN
        TP_percent = (TP / total_elements) * 100
        FP_percent = (FP / total_elements) * 100
        TN_percent = (TN / total_elements) * 100
        FN_percent = (FN / total_elements) * 100

        # Append bootstrapped results to lists
        bootstrapped_TPs.append(TP_percent)
        bootstrapped_FPs.append(FP_percent)
        bootstrapped_TNs.append(TN_percent)
        bootstrapped_FNs.append(FN_percent)
    nbins=70
    # Plot histograms and fit Gaussians
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # True Positive
    axs[0, 0].hist(bootstrapped_TPs, bins=nbins, alpha=0.5, color='blue', density=True, label='Data')
    axs[0, 0].axvline(conf[0][0],color='black',linestyle='dashed',label='True Mean')
    mu, std = norm.fit(bootstrapped_TPs)
    xmin, xmax = axs[0, 0].get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    axs[0, 0].plot(x, p, 'k', linewidth=2,label='Gaussian Fit')
    axs[0, 0].set_title('True Positive')
    axs[0, 0].legend()
    
    # False Positive
    axs[0, 1].hist(bootstrapped_FPs, bins=nbins, alpha=0.5, color='red', density=True, label='Data')
    axs[0, 1].axvline(conf[0][1],color='black',linestyle='dashed', label='True Mean')
    mu, std = norm.fit(bootstrapped_FPs)
    xmin, xmax = axs[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    axs[0, 1].plot(x, p, 'k', linewidth=2,label='Gaussian Fit')
    axs[0, 1].set_title('False Positive')
    axs[0, 1].legend()
    
    # False Negative
    axs[1, 0].hist(bootstrapped_FNs, bins=nbins, alpha=0.5, color='orange', density=True, label='Data')
    axs[1, 0].axvline(conf[1][0],color='black',linestyle='dashed', label='True Mean')
    mu, std = norm.fit(bootstrapped_FNs)
    xmin, xmax = axs[1, 0].get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    axs[1, 0].plot(x, p, 'k', linewidth=2,label='Gaussian Fit')
    axs[1, 0].set_title('False Negative')
    axs[1, 0].legend()
    
    # True Negative
    axs[1, 1].hist(bootstrapped_TNs, bins=nbins, alpha=0.5, color='green', density=True, label='Data')
    axs[1, 1].axvline(conf[1][1],color='black',linestyle='dashed', label='True Mean')
    mu, std = norm.fit(bootstrapped_TNs)
    xmin, xmax = axs[1, 1].get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    axs[1, 1].plot(x, p, 'k', linewidth=2,label='Gaussian Fit')
    axs[1, 1].set_title('True Negative')
    axs[1, 1].legend()
    


    for ax in axs.flat:
        ax.set(xlabel='Percentage', ylabel='Frequency')


    plt.tight_layout()
    plt.show()

    # Return bootstrapped distributions
    return bootstrapped_TPs, bootstrapped_FPs,bootstrapped_FNs,bootstrapped_TNs


test=compare_forecastsBoot(df_sym_forecastsM, df_real_symsM,-10,percentagesM,1000)
#prints every 100th 'bootstrapping'. Is quite slow. I ran for 100,000
#%%
#Prints stats
#Prop. diff in mean is just how far off the sample mean is from the true mean as a fraction of the true mean
#Shows how representative our sample is
nameys=["TP","FP","FN","TN"]
for i in range(len(test)):
    print(nameys[i]+": Mean = " +str(round(np.mean(test[i]),4))+", std = "+str(round(np.std(test[i]),4)))
    print("Proportional Variation, σ/μ : "+str(round((np.std(test[i])/(np.mean(test[i]))),5)))
    print("Prop. diff. in True and Sample Mean: "+str(round((np.mean(test[i])-np.array(percentagesM).flatten()[i])/np.mean(test[i]),5)))
    print()
    
#%%   
import time

# Function to generate bootstrapped means
def bootstrap(data, n_bootstraps=1000):
    means = []
    
    # Perform bootstrapping
    for _ in range(n_bootstraps):
        # Resample the data with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        means.append(bootstrap_mean)
    
    # Calculate the variance of the means
    mean = np.mean(means)
    std = np.std(means)
    
    return mean, std, means
#allSSC=np.mean(np.array([zvCCsA, zvCCsD, zvCCsW]),axis=0)
#Different ways of defining what the mean single spacecraft is
allSSC=np.array([zvCCsA, zvCCsD, zvCCsW]).ravel()
#%%
# Generate data
nIter = 10000
#vals = [zvCCsA, zvCCsD, zvCCsW, zvCCsM]
#meths=["ACE","DSCOVR","Wind","Multi"]

vals = [zvCCsA, zvCCsD, zvCCsW, zvCCsM,allSSC]
meths=["ACE","DSCOVR","Wind","Multi","Mean Single"]

bigres = []

start_time = time.time()

for i in range(len(vals)):
    results = bootstrap(vals[i], nIter)
    bigres.append(results)
    print(i)

end_data_generation_time = time.time()
#%%
# Plotting
plt.figure()

for i in range(len(vals)):
    plt.hist(bigres[i][2], bins=100, density=True, alpha=0.4, label=meths[i])
    mu, std = norm.fit(bigres[i][2])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=0.5)
    plt.xlabel('Mean Zero Value CC')
    plt.ylabel('Probability Density')

for i in range(len(vals)):
    if i == 0:
        plt.axvline(np.mean(vals[i]), color='black', linestyle='dashed', label='True Means')
    else:
        plt.axvline(np.mean(vals[i]), color='black', linestyle='dashed')
plt.grid(alpha=0.4)
plt.legend()
plt.show()

end_plotting_time = time.time()

# Calculate the elapsed time
elapsed_data_generation_time = end_data_generation_time - start_time
elapsed_plotting_time = end_plotting_time - end_data_generation_time
total_elapsed_time = end_plotting_time - start_time

print("Data generation time:", elapsed_data_generation_time, "seconds for " + str(nIter) + " iterations")
print("Plotting time:", elapsed_plotting_time, "seconds")
print("Total elapsed time:", total_elapsed_time, "seconds")

#%%
for i in range(len(vals)):
    print(meths[i] + ": Av. Mean = "+str(bigres[i][0]) + ", std = " +str(round(bigres[i][1],4)))
    print("Prop. diff. in True and Sample Mean: "+str(round((bigres[i][0]-np.mean(vals[i]))/np.mean(vals[i]),6)))
    print()

#%%
#Prints the true means of each method for comparison
for i in range(len(vals)):
    print(round(np.mean(vals[i]),4))
    
#%%
#Doing exaaaaaactly the same but with just multi and single spacecraft
meths=["Multi","Mean Single"]
vals=[zvCCsM,allSSC]
bigres = []

start_time = time.time()
nIter=1000000
for i in range(len(vals)):
    results = bootstrap(vals[i], nIter)
    bigres.append(results)
    print(i)
#%%
end_data_generation_time = time.time()
plt.figure()

for i in range(len(vals)):
    plt.hist(bigres[i][2], bins=100, density=True, alpha=0.4, label=meths[i])
    mu, std = norm.fit(bigres[i][2])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=0.5)
    plt.xlabel('Mean Zero Value CC')
    plt.ylabel('Probability Density')

for i in range(len(vals)):
    if i == 0:
        plt.axvline(np.mean(vals[i]), color='black', linestyle='dashed', label='True Means')
    else:
        plt.axvline(np.mean(vals[i]), color='black', linestyle='dashed')
plt.grid(alpha=0.4)
plt.legend()
plt.show()

end_plotting_time = time.time()

# Calculate the elapsed time
elapsed_data_generation_time = end_data_generation_time - start_time
elapsed_plotting_time = end_plotting_time - end_data_generation_time
total_elapsed_time = end_plotting_time - start_time

print("Data generation time:", elapsed_data_generation_time, "seconds for " + str(nIter) + " iterations")
print("Plotting time:", elapsed_plotting_time, "seconds")
print("Total elapsed time:", total_elapsed_time, "seconds")

#%%
for i in range(len(vals)):
    print(meths[i] + ": Av. Mean = "+str(bigres[i][0]) + ", std = " +str(round(bigres[i][1],4)))
    print("Prop. diff. in True and Sample Mean: "+str(round((bigres[i][0]-np.mean(vals[i]))/np.mean(vals[i]),6)))
    print()

#%%
#Getting probability that multi is better than mean single
gauss1=list(bigres[0][:2])
gauss2=list(bigres[1][:2])
def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x=np.linspace(-0.03,0.08,10000)
gauss3=gaussian(x,gauss1[0]-gauss2[0],np.sqrt(gauss1[1]**2+gauss2[1]**2))
plt.plot(x,gauss3)
plt.xlabel("$CC_{Multi} - CC_{Mean}$")
plt.ylabel("PDF")
plt.fill_between(x, gauss3, where=(x < 0), color='red', alpha=0.3)
plt.fill_between(x, gauss3, where=(x >= 0), color='green', alpha=0.3)
plt.grid(alpha=0.5)
probability_greater_than_zero = 1 - norm.cdf(0, loc=gauss3[0], scale=gauss3[1])

print("Probability that the value is greater than 0:", probability_greater_than_zero)#%%

#%%
df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')
split_times = np.load('split_data_times.npy')

