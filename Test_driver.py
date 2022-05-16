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
dispatch = 'SLP'
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

output_dir13 = pathlib.Path(output_dir12).joinpath(dispatch)
if not os.path.isdir(output_dir13):
    os.mkdir(output_dir13)

batSizes = [0, 100, 300]
pvSizes = [0, 50, 100]

# voltage limits
vmin = 0.97
vmax = 1.03

for ba, batSize in enumerate(batSizes): 
    for pv, pvSize in enumerate(pvSizes):
        
        output_dir1 = pathlib.Path(output_dir13).joinpath(f"bat_{ba}_pv_{pv}")
        if not os.path.isdir(output_dir1):
            os.mkdir(output_dir1)
            
        ####################################
        # First thing: compute the initial Dispatch
        ####################################
        demandProfile, LMP, OperationCost, mOperationCost = SLP_LP_scheduling(batSize, pvSize, output_dir1, vmin, vmax, userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)
                
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
        # LMP index random
        LMP_index = np.random.choice(LMP_size, LMP_size, replace=False)

        # create smart charging driver object
        char_obj = smartCharging_driver(ext, arrivalTime_list, departureTime_list, initEnergy_list, evCapacity_list, initW_list) 
        
        # define folder to store smartcharging results
        output_dir = pathlib.Path(output_dir1).joinpath("SmartCharging")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            
        ####################################
        # Second thing: compute the decentralized smart charging
        ####################################
        
        # define number of iterations
        iterations = 10 
        evh = [60, 70, -1]
        evh_size = len(evh)
        
        # number of EV loop
        for ev in range(evh_size): 
            # initialize store variables as lists
            prevLMP = LMP.copy() 
            prevDemand = demandProfile.copy()
            OpCost_list = list()
            mOpCost_list = list()
        
            # save base case LMP 
            LMP_list.append(LMP) 
            OpCost_list.append(OperationCost) 
            mOpCost_list.append(mOperationCost) 
        
            # create a folder to store LMP per EV
            output_dirEV = pathlib.Path(output_dir).joinpath(f"EV_{ev}")
            
            if not os.path.isdir(output_dirEV):
                os.mkdir(output_dirEV)
                
            # per node LMP comparison
            dLMP_node_L2 = np.zeros((len(LMP.index), iterations))
            dLMP_node_L1 = np.zeros((len(LMP.index), iterations))
            dLMP_node_Linf = np.zeros((len(LMP.index), iterations))

            # prelocate demand difference variable
            diffDemand = np.zeros(demandProfile.shape)

            # iterations loop
            for it in range(iterations): 
                
                # keep track
                print(f"bat:{ba}_pv:{pv}_EV:{ev}_it:{it}")
        
                # compute EV corrected demand
                newDemand = char_obj.charging_driver(output_dirEV, it, demandProfile, prevLMP, LMP_index, plot)
                
                # compute scheduling with new demand
                _, LMP_EVC, OperationCost_EVC, mOperationCost_EVC = SLP_LP_scheduling(batSize, pvSize, output_dirEV, vmin, vmax, userDemand=newDemand, plot=True, freq="30min", dispatchType=dispatch)
        
                # store corrected values 
                OpCost_list.append(OperationCost_EVC) 
                mOpCost_list.append(mOperationCost_EVC) 
                
                # store node-base LMP difference
                dLMP_node_L2[:,it] = np.linalg.norm(LMP_EVC - prevDemand, ord=2, axis=1)
                dLMP_node_L1[:,it] = np.linalg.norm(LMP_EVC - prevDemand, ord=1, axis=1)
                dLMP_node_Linf[:,it] = np.linalg.norm(LMP_EVC - prevDemand, ord=np.inf, axis=1)

                # Store demand difference
                diffDemand = newDemand - prevDemand
                if plot:
                    diffDemand_pd = pd.DataFrame(diffDemand)
                    fig, ax = plt.subplots()
                    sns.heatmap(diffDemand, annot=False)
                    fig.tight_layout()

                    output_dirDemand = pathlib.Path(output_dirEV).joinpath("Demand")
                    if not os.path.isdir(output_dirDemand):
                        os.mkdir(output_dirDemand)

                    output_img = pathlib.Path(output_dirDemand).joinpath(f"Demand_diff_it_{it}"+ ext)
                    plt.savefig(output_img)
                    plt.close('all')

                # assign new definition
                prevLMP = LMP_EVC.copy() 
                prevDemand = newDemand.copy()

            ##########
            # node based LMP difference (L2)
            pd_dLMP = pd.DataFrame(dLMP_node_L2)
            dLMP_file = pathlib.Path(output_dirEV).joinpath("dLMP_node_L2.pkl")
            pd_dLMP.to_pickle(dLMP_file)
            # plot
            plt.clf()
            fig, ax = plt.subplots()
            sns.heatmap(pd_dLMP, annot=False)
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("dLMP_node_L2"+ ext)
            plt.savefig(output_img)
            plt.close('all')
            
            # node base LMP difference
            pd_dLMP = pd.DataFrame(dLMP_node_L1)
            dLMP_file = pathlib.Path(output_dirEV).joinpath("dLMP_node_L1.pkl")
            pd_dLMP.to_pickle(dLMP_file)
            # plot
            plt.clf()
            fig, ax = plt.subplots()
            sns.heatmap(pd_dLMP, annot=False)
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("dLMP_node_L1"+ ext)
            plt.savefig(output_img)
            plt.close('all')
            
            # node base LMP difference (Linf)
            pd_dLMP = pd.DataFrame(dLMP_node_Linf)
            dLMP_file = pathlib.Path(output_dirEV).joinpath("dLMP_node_Linf.pkl")
            pd_dLMP.to_pickle(dLMP_file)
            # plot
            plt.clf()
            fig, ax = plt.subplots()
            sns.heatmap(pd_dLMP, annot=False)
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("dLMP_node_Linf"+ ext)
            plt.savefig(output_img)
            plt.close('all')
        
            ########################
            ### costs
            plt.clf()
            fig, ax = plt.subplots()
            plt.ylim(bottom=min(OpCost_list))
            plt.stem(OpCost_list - min(OpCost_list))
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("Operation_cost_list"+ ext)
            plt.savefig(output_img)
            plt.close('all')

            plt.clf()
            fig, ax = plt.subplots()
            plt.ylim(bottom=min(OpCost_list[1:]))
            plt.stem(OpCost_list[1:] - min(OpCost_list[1:]))
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("Operation_cost_list_1"+ ext)
            plt.savefig(output_img)
            plt.close('all')

            ########################
            ###  mcosts
            plt.clf()
            fig, ax = plt.subplots()
            plt.ylim(bottom=min(mOpCost_list))
            plt.stem(mOpCost_list)
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("mOperation_cost_list"+ ext)
            plt.savefig(output_img)
            plt.close('all')

            plt.clf()
            fig, ax = plt.subplots()
            plt.ylim(bottom=min(mOpCost_list[1:]))
            plt.stem(mOpCost_list[1:] - min(mOpCost_list[1:]))
            fig.tight_layout()
            output_img = pathlib.Path(output_dirEV).joinpath("mOperation_cost_list_1"+ ext)
            plt.savefig(output_img)
            plt.close('all')
