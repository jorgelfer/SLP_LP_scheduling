# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import SLP_LP_scheduling
from Methods.smartCharging_driver import smartCharging_driver 
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import time
import seaborn as sns

ext = '.png'
dispatch = 'LP'
plot = False

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

output_dir = pathlib.Path(output_dir12).joinpath(dispatch)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

batSize = 300
pvSize = 100

# voltage limits
vmin = 0.97
vmax = 1.03

####################################
# First thing: compute the initial Dispatch
####################################
demandProfile, LMP, OperationCost, mOperationCost = SLP_LP_scheduling(batSize, pvSize, output_dir, vmin, vmax, userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)

# define smartcharging settings
# Set random seed so results are repeatable
np.random.seed(2022) 
# define init energy 
LMP_size = np.size(LMP,0)
initEnergy_list = [np.random.uniform(18, 70) for i in range(LMP_size)]
# define ev capacity
evCapacity_list = [np.random.uniform(81.5, 89.5) for i in range(LMP_size)]
# define arrival time
arrivalTime_list = [f"{np.random.randint(16, 22)}:{np.random.randint(0,2)*3}0" for i in range(LMP_size)]
# define departure time
departureTime_list = [f"{np.random.randint(6, 12)}:{np.random.randint(0,2)*3}0" for i in range(LMP_size)]
# create weights using dirichlet distribution: the sumation add up to 1
initW_list = [np.random.dirichlet(np.ones(4),size=1) for i in range(LMP_size)]

# create smart charging driver object
char_obj = smartCharging_driver(ext, arrivalTime_list, departureTime_list, initEnergy_list, evCapacity_list, initW_list) 

# define the number of EV to analyze
evh = [94, 80, 65, 50, 35, 20, 5, 0]
evh_size = len(evh)

# prelocate variable to store results
iterations = 5 
folds = 5 
opCost =  np.zeros((folds, iterations, evh_size))
mopCost = np.zeros((folds, iterations, evh_size))

# number of EV loop
for ev in range(evh_size):
    
    # LMP index random
    try:
        LMP_index = np.random.choice(LMP_size, size=(folds,evh[ev]), replace=False)
    except:
        LMP_index = np.random.choice(LMP_size, size=(folds,evh[ev]), replace=True)
        
    # iterations loop
    for f in range(folds): 

        prevLMP = LMP.copy()
    
        # number of folds
        for it in range(iterations):

            # keep track
            print(f"EV:{ev}_fold:{f}_it:{it}")
        
            ####################################
            # Second thing: compute the decentralized smart charging using all EV
            ####################################
            newDemand = char_obj.charging_driver(output_dir, evh[ev], demandProfile, prevLMP, LMP_index[f], plot)
            
            ####################################
            # Third Thing: compute dispatch including EV
            ####################################
            _, LMP_EVC, OperationCost_EVC, mOperationCost_EVC = SLP_LP_scheduling(batSize, pvSize, output_dir, vmin, vmax, userDemand=newDemand, plot=plot, freq="30min", dispatchType=dispatch)

            # store
            opCost[f, it, ev] = OperationCost_EVC
            mopCost[f, it, ev] = mOperationCost_EVC
            
            # update LMP
            prevLMP = LMP_EVC.copy()

##########
# plot
for it in range(iterations): 
    # node based LMP difference (L2)
    pd_OpCost = pd.DataFrame(opCost[:,it,:], columns=np.asarray(evh))
    OpCost_file = pathlib.Path(output_dir).joinpath(f"OpCost_{it}.pkl")
    pd_OpCost.to_pickle(OpCost_file)
    # plot
    plt.clf()
    fig, ax = plt.subplots()
    pd_OpCost.boxplot()
    fig.tight_layout()
    output_img = pathlib.Path(output_dir).joinpath(f"OpCost_boxplot_{it}"+ ext)
    plt.savefig(output_img)
    plt.close('all')
    
    # node based LMP difference (L2)
    pd_OpCost = pd.DataFrame(mopCost[:,it,:], columns=np.asarray(evh))
    OpCost_file = pathlib.Path(output_dir).joinpath(f"mOpCost_{it}.pkl")
    pd_OpCost.to_pickle(OpCost_file)
    # plot
    plt.clf()
    fig, ax = plt.subplots()
    pd_OpCost.boxplot()
    fig.tight_layout()
    output_img = pathlib.Path(output_dir).joinpath(f"mOpCost_boxplot_{it}"+ ext)
    plt.savefig(output_img)
    plt.close('all')
