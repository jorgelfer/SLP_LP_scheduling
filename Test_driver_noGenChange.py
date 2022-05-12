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
Ope_file = pathlib.Path(output_dir1).joinpath("OperationCost.npy")

if compute:
    demandProfile, LMP, OperationCost, mOperationCost = SLP_LP_scheduling(output_dir1, userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)
    # save
    demandProfile.to_pickle(demand_file)
    LMP.to_pickle(LMP_file)
    with open(Ope_file, 'wb') as wb:
        np.save(wb, OperationCost)
        np.save(wb, mOperationCost)
else:
    # load
    demandProfile = pd.read_pickle(demand_file)
    LMP = pd.read_pickle(LMP_file)
    with open(Ope_file, 'rb') as rb:
        OperationCost = np.load(rb)
        mOperationCost = np.load(rb)

# compute total demand
totalDemand =  demandProfile.sum(axis = 0).to_frame()

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
initW = [np.random.dirichlet(np.ones(4),size=1) for i in range(LMP_size)]

output_dir = pathlib.Path(output_dir1).joinpath("SmartCharging")

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
####################################
# Second thing,
# compute the decentralized smart charging
####################################

# define number of iterations
iterations = 5
evh = [10, 40, 70, -1]
LMP_size = len(evh)

# create a convergence variable to store results
dLMP_F = np.zeros((LMP_size, iterations))
dLMP_N = np.zeros((LMP_size, iterations))
dLMP_L1 = np.zeros((LMP_size, iterations))
dLMP_L2 = np.zeros((LMP_size, iterations))
dLMP_Linf = np.zeros((LMP_size, iterations))

OpCost = np.zeros((LMP_size, iterations))
mOpCost = np.zeros((LMP_size, iterations))

# number of EV loop
for ev in range(LMP_size): 
    # initialize store variables as lists
    LMP_list = list()
    OpCost_list = list()
    mOpCost_list = list()

    # save base case LMP 
    LMP_list.append(LMP) 
    OpCost_list.append(OperationCost) 
    mOpCost_list.append(mOperationCost) 

    # create a folder to store LMP
    output_dirEV = pathlib.Path(output_dir).joinpath(f"EV_{ev}")
    
    if not os.path.isdir(output_dirEV):
        os.mkdir(output_dirEV)

    # iterations loop
    for it in range(iterations): 
        
        # keep track
        print(f"{ev}-{it}")

        # smart charging module
        EV_demandProfile = np.zeros(demandProfile.shape)
        EV_demandProfile = pd.DataFrame(EV_demandProfile, index=LMP.index, columns=LMP.columns)
        
        # create smart charging object
        charging_obj = SmartCharging(numberOfHours=24, pointsInTime=LMP.shape[1]) 
        
        for i, ind in enumerate(LMP.index[:evh[ev]]):
            #individual LMP (pi)
            pi = LMP_list[-1].loc[ind,:]
            # reorder dates
            pi = reorder_dates(pi)
            # transform to array
            pi = np.expand_dims(pi.values, axis=1)
            # household demand profile
            PH = demandProfile.loc[ind,:] # normal demand
            # reorder dates
            PH = reorder_dates(PH)
            PH = np.expand_dims(PH.values, axis=1)
            # user defined weights
            w = np.squeeze(initW[i]) 
            #EV initial conditions
            arrTime = arrivalTime_list[i]
            depTime = departureTime_list[i]
            initEnergy = initEnergy_list[i]
            evCapacity = evCapacity_list[i] 

            # optimal EV charging using the smart charging object
            PV_star,_,_,_ = charging_obj.QP_charging(pi, PH, w, arrTime, depTime, initEnergy, evCapacity) # user specific values
            # reorder index from dataSeries:
            PV_star = order_dates(PV_star, freq="30min")
            # assign to the profile
            EV_demandProfile.loc[ind,:] = PV_star
            
        ####################################
        # Third thing,
        # compute dispatch with decentralized smart charging correction
        ####################################
        #define new demand
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
            output_img = pathlib.Path(output_dirEV).joinpath(f"EVcorrected_demand_{it}"+ ext)
            plt.savefig(output_img)
            plt.close('all')
        
        # compute scheduling
        _, LMP_EVC, OperationCost_EVC, mOperationCost_EVC = SLP_LP_scheduling(output_dirEV, userDemand=newDemand, plot=True, freq="30min", dispatchType=dispatch)

        # store corrected values 
        LMP_list.append(LMP_EVC)
        OpCost_list.append(OperationCost_EVC) 
        mOpCost_list.append(mOperationCost_EVC) 
        
        # store difference
        dLMP_F[ev, it] = np.linalg.norm(LMP_list[it]-LMP_list[it+1], 'fro')
        dLMP_N[ev, it] = np.linalg.norm(LMP_list[it]-LMP_list[it+1], 'nuc')
        dLMP_L1[ev, it] = np.linalg.norm(LMP_list[it]-LMP_list[it+1], 1)
        dLMP_L2[ev, it] = np.linalg.norm(LMP_list[it]-LMP_list[it+1], 2)
        dLMP_Linf[ev, it] = np.linalg.norm(LMP_list[it]-LMP_list[it+1], np.inf)

        # store operation cost difference
        OpCost[ev, it] = abs(OpCost_list[it] - OpCost_list[it+1])
        
        # store operation cost
        mOpCost[ev, it] = abs(mOpCost_list[it] - mOpCost_list[it+1])

###############
# save matrices
# F norm
pd_dLMP = pd.DataFrame(dLMP_F)
dLMP_file = pathlib.Path(output_dir).joinpath("dLMP_F.pkl")
pd_dLMP.to_pickle(dLMP_file)
# plot
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(pd_dLMP, annot=False)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("dLMP_F.png")
plt.savefig(output_img)
plt.close('all')

# N norm
pd_dLMP = pd.DataFrame(dLMP_N)
dLMP_file = pathlib.Path(output_dir).joinpath("dLMP_N.pkl")
pd_dLMP.to_pickle(dLMP_file)
# plot
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(pd_dLMP, annot=False)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("dLMP_N.png")
plt.savefig(output_img)
plt.close('all')

# L1 norm
pd_dLMP = pd.DataFrame(dLMP_L1)
dLMP_file = pathlib.Path(output_dir).joinpath("dLMP_L1.pkl")
pd_dLMP.to_pickle(dLMP_file)
# plot
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(pd_dLMP, annot=False)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("dLMP_L1.png")
plt.savefig(output_img)
plt.close('all')

# L2 norm
pd_dLMP = pd.DataFrame(dLMP_L2)
dLMP_file = pathlib.Path(output_dir).joinpath("dLMP_L2.pkl")
pd_dLMP.to_pickle(dLMP_file)
# plot
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(pd_dLMP, annot=False)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("dLMP_L2.png")
plt.savefig(output_img)
plt.close('all')

# Linf norm
pd_dLMP = pd.DataFrame(dLMP_Linf)
dLMP_file = pathlib.Path(output_dir).joinpath("dLMP_Linf.pkl")
pd_dLMP.to_pickle(dLMP_file)
# plot
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(pd_dLMP, annot=False)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("dLMP_Linf.png")
plt.savefig(output_img)
plt.close('all')

########################
### costs
pd_OpCost = pd.DataFrame(OpCost)
OpCost_file = pathlib.Path(output_dir).joinpath("OpCost.pkl")
pd_OpCost.to_pickle(OpCost_file)
# plot
plt.clf()
fig, ax = plt.subplots()
pd_cost = pd.DataFrame(OpCost)
sns.heatmap(pd_cost, annot=False, vmin=2e6, vmax=2.30e6)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("Operation_cost"+ ext)
plt.savefig(output_img)
plt.close('all')

##############
pd_mOpCost = pd.DataFrame(mOpCost)
mOpCost_file = pathlib.Path(output_dir).joinpath("mOpCost.pkl")
pd_mOpCost.to_pickle(mOpCost_file)
# plot
plt.clf()
fig, ax = plt.subplots()
pd_mcost = pd.DataFrame(mOpCost)
sns.heatmap(pd_mcost, annot=False, vmin=2e6, vmax=2.30e6)
fig.tight_layout()
output_img = pathlib.Path(output_dir).joinpath("mOperation_cost"+ ext)
plt.savefig(output_img)
plt.close('all')
