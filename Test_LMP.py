# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import SLP_LP_scheduling
from Methods.SmartCharging import SmartCharging 
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import time
import seaborn as sns
import time

def reorder_dates(pdSeries):
    pdSeries1 = pdSeries[:pdSeries.index.get_loc('08:00')]
    pdSeries2 = pdSeries[pdSeries.index.get_loc('08:00'):]
    pdSeries = pd.concat((pdSeries2, pdSeries1))
    return pdSeries

def order_dates(array, freq="30min"):
    # initialize time pdSeries
    pdSeries = pd.Series(np.zeros(len(array)))
    pdSeries.index = pd.date_range("00:00", "23:59", freq=freq).strftime('%H:%M')
    # reorder init dataSeries to match Kartik's order
    pdSeries = reorder_dates(pdSeries)
    # assign obtained values
    pdframe = pdSeries.to_frame()
    pdframe[0] = array
    pdSeries = pdframe.squeeze()
    # order back to normal 
    pdSeries1 = pdSeries[:pdSeries.index.get_loc('00:00')]
    pdSeries2 = pdSeries[pdSeries.index.get_loc('00:00'):]
    pdSeries = pd.concat((pdSeries2, pdSeries1))
    return pdSeries

ext = '.png'
dispatch = 'LP'

script_path = os.path.dirname(os.path.abspath(__file__))

# output directory
# time stamp 
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)  
# create directory to store results
today = time.strftime('%b-%d-%Y', t)
directory = "Results_" + today
output_dir12 = pathlib.Path(script_path).joinpath("outputs", directory)

if not os.path.isdir(output_dir12):
    os.mkdir(output_dir12)

output_dir1 = pathlib.Path(output_dir12).joinpath(dispatch)
if not os.path.isdir(output_dir1):
    os.mkdir(output_dir1)

####################################
# First thing,
# compute the initial Dispatch
####################################

compute = True 
plot = True 
# initial scheduling results
demand_file = pathlib.Path(output_dir1).joinpath("demandProfile.pkl")
LMP_file = pathlib.Path(output_dir1).joinpath("LMP.pkl")

if compute:
    demandProfile, LMP = SLP_LP_scheduling(userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)
    # save
    demandProfile.to_pickle(demand_file)
    LMP.to_pickle(LMP_file)
    
else:
    # load
    demandProfile = pd.read_pickle(demand_file)
    LMP = pd.read_pickle(LMP_file)

# compute total demand
totalDemand =  demandProfile.sum(axis = 0).to_frame()

EV_demandProfile = np.zeros(demandProfile.shape)
EV_demandProfile = pd.DataFrame(EV_demandProfile, index=demandProfile.index, columns=demandProfile.columns)
EV_demandProfile.iloc[0,40] += 200 
print(demandProfile.index[0])
newDemand = demandProfile + EV_demandProfile

if plot:
    # new demand plot
    plt.clf()
    fig, ax = plt.subplots()
    
    totalNewDemand = newDemand.sum(axis = 0).to_frame()
    concat3 = pd.concat([totalDemand, totalNewDemand], axis=1)
    concat3.plot()
    plt.legend(['load_toalDemand', 'load_EV_totalDemand'], prop={'size': 10})
        
    fig.tight_layout()
    output_img = pathlib.Path(output_dir1).joinpath("EVcorrected_demand"+ ext)
    plt.savefig(output_img)
    plt.close('all')

time.sleep(50) # Sleep for 3 secondso

# compute scheduling
_, LMP_EVC = SLP_LP_scheduling(userDemand=newDemand, plot=plot, freq="30min", dispatchType=dispatch)

