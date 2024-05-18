# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import SLP_LP_scheduling
from Methods.schedulingDriver import schedulingDriver
from Methods.smartCharging_driver import smartCharging_driver 
import py_dss_interface
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import time
import seaborn as sns
import shutil
from functools import reduce
import json

ext = '.png'
dispatch = 'SLP'
metric = np.inf# 1,2,np.inf
plot = False 
h = 6 
w = 4 

# script_path = os.path.dirname(os.path.abspath(__file__))
DIR = os.getcwd()

# output directory
# time stamp 
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)  
# create directory to store results
today = time.strftime('%b-%d-%Y', t)
directory = "Results2_" + today

output_dir12 = os.path.join(DIR, "outputs", directory)

if not os.path.isdir(output_dir12):
    os.mkdir(output_dir12)

output_dir13 = os.path.join(output_dir12, dispatch)
if not os.path.isdir(output_dir13):
    os.mkdir(output_dir13)
    
output_dir14 = os.path.join(output_dir13, f"L_{metric}")
if not os.path.isdir(output_dir14):
    os.mkdir(output_dir14)
else:
    shutil.rmtree(output_dir14)
    os.mkdir(output_dir14)

batSizes = [0]
pvSizes = [0]

# voltage limits
vmin = 0.95
vmax = 1.05

# define DSS path
dataset = "IEEETestCases"
NetworkModel = "123Bus_wye" # "SecondaryTestCircuit_modified", "13Bus", "123Bus", "case3", "4Bus-DY-Bal"
InFile1 = "IEEE123Master.dss" # "Master.DSS", "IEEE13Nodeckt.dss", "IEEE123Master.dss", "case3_unbalanced.dss", "4Bus-DY-Bal.dss"

# Opening JSON file
json_path = os.path.join(DIR, "..", dataset, NetworkModel, "qsts.json")
f = open(json_path)
 
# returns JSON object as 
# a dictionary
qsts = json.load(f)
time_vec = qsts["time"]
PointsInTime = len(qsts["time"])
outDSS = dict()

####
# preprocess load
###
nodes = qsts["dpdp"]["nodes"]
pdict = {key: np.zeros(PointsInTime) for key in nodes}
qdict = {key: np.zeros(PointsInTime) for key in nodes}
ldict = {}
for load in qsts["load"]:

    # get load bus uid
    bus = load["bus"] 

    # get load phases
    phases = load["phases"]

    # load power 
    for ph in phases:
        pdict[bus + f".{ph}"] = np.asarray(load["p"][f"{ph}"]) 
        qdict[bus + f".{ph}"] = np.asarray(load["q"][f"{ph}"]) 
        ldict[bus + f".{ph}"] = load["uid"]

dfDemand = pd.DataFrame(np.stack([pdict[n] for n in nodes]), index = np.asarray(nodes)) # in kW
dfDemandQ = pd.DataFrame(np.stack([qdict[n] for n in nodes]), index = np.asarray(nodes)) # in kW
loadNames = pd.Series(ldict)
outDSS['initDemand'] = dfDemand
outDSS['initDemandQ'] = dfDemandQ
outDSS['loadNames'] = loadNames

####
# preprocess initial flows 
####
bpns = qsts["dpdp"]["bpns"]
fdict = {key: np.zeros(PointsInTime) for key in bpns}
for br in qsts["branch"]:
    # get branch uid
    uid = br["uid"].split(".")[1]

    # for each flow
    for ph in br["phases"]:
        lenp = len(br["p_nm"][f"{ph}"])
        fdict[uid + f".{ph}"] = np.asarray(br["p_nm"][f"{ph}"])
Pjk_0 = pd.DataFrame(np.stack([fdict[n] for n in bpns]), index = np.asarray(bpns))
outDSS['initPjks'] = Pjk_0


####
# preprocess initial voltages magnitutes
####
nodes = qsts["dpdp"]["nodes"]
vdict = {key: np.zeros(PointsInTime) for key in nodes}
for bus in qsts["bus"]:

    # get load bus uid
    uid = bus["uid"] 

    # get load phases
    phases = bus["phases"]

    # load power 
    for ph in phases:
        vdict[uid + f".{ph}"] = np.asarray(bus["vm"][f"{ph}"])
Vm_0 = pd.DataFrame(np.stack([vdict[n] for n in nodes]), index = np.asarray(nodes))
outDSS['initVolts'] = Vm_0


####
# initial generation
####
nodes = qsts["dpdp"]["nodes"]
gdict = {key: np.zeros(PointsInTime) for key in nodes}
for vs in qsts["vsource"]:

    # get load bus uid
    uid = vs["bus"] 

    # get load phases
    phases = vs["phases"]

    # load power 
    for ph in phases:
        gdict[uid + f".{ph}"] = np.asarray(vs["p"][f"{ph}"])

Pg_0 = pd.DataFrame(np.stack([gdict[n] for n in nodes]), index = np.asarray(nodes))
outDSS['initPower'] = Pg_0


####
# preprocess voltage base for each node
####
nodes = qsts["dpdp"]["nodes"]
vdict = {key: 0.0 for key in nodes}
for bus in qsts["bus"]:
    # get bus uid
    uid = bus["uid"]

    # get load phases
    phases = bus["phases"]

    # load power 
    for ph in phases:
        vdict[uid + f".{ph}"] = bus["kV_base"]

v_basei = pd.DataFrame(np.asarray([vdict[n] for n in nodes]), index = np.asarray(nodes))
v_base = np.kron(v_basei, np.ones((1, PointsInTime)))
v_base = pd.DataFrame(v_base, index=v_basei.index)
outDSS['nodeBaseVolts'] = v_base 


####
# preprocess PTDF
###
# row length
rl = len(qsts["dpdp"]["bpns"])

# column length
cl = len(qsts["dpdp"]["nodes"])

# reshape flatten array
PTDF = np.reshape(qsts["dpdp"]["matrix_ij"], (rl, cl), order='F')
PTDF = pd.DataFrame(PTDF, columns=qsts["dpdp"]["nodes"], index=qsts["dpdp"]["lns"])
outDSS['PTDF'] = PTDF


####
# preprocess voltage sensitivity
####
# row and columng length
rcl = len(qsts["dvdp"]["nodes"])

# reshape flatten array
dvdp = np.reshape(qsts["dvdp"]["matrix"], (rcl, rcl), order='F')
dvdp = pd.DataFrame(dvdp, columns=qsts["dvdp"]["nodes"], index=qsts["dvdp"]["nodes"])
outDSS['dvdp'] = dvdp

####
# preprocess branches
####

# line costs
Pijcost = 0.0 * np.zeros((len(PTDF), PointsInTime))
clin = np.reshape(Pijcost.T, (1,Pijcost.size), order="F")

# define line limits
bpns = qsts["dpdp"]["bpns"]
ldict = {key: 0.0 for key in bpns}

for br in qsts["branch"]:
    # get branch uid
    uid = br["uid"].split(".")[1]

    # assign normal flow limit (already comes by phase in kW)
    normal_flow_limit = br["normal_flow_limit"] # in kVA

    for ph in br["phases"]:
        ldict[uid + f".{ph}"] = normal_flow_limit

# Lmaxi = pd.DataFrame(np.asarray([ldict[n] for n in bpns]), np.asarray(bpns))
Lmaxi = 2000 * np.ones((len(PTDF),1))
Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
Lmax = pd.DataFrame(Lmax, index=np.asarray(bpns))
outDSS['Pjk_lim'] = Lmax

# # optimization
# for ba, batSize in enumerate(batSizes): 
#     for pv, pvSize in enumerate(pvSizes):
#         #Energy scheduling driver function   
#         outGen, outDR, outPchar, outPdis, outLMP, costPdr, cgn, mobj = schedulingDriver(batSize, pvSize, DIR, 'Dispatch', 'h', DIR, InFile1, outDSS, dispatch, vmin, vmax, plot=plot)
#         # outES = save_ES(script_path, outGen, outDR, outPchar, outPdis)


for ba, batSize in enumerate(batSizes): 
    for pv, pvSize in enumerate(pvSizes):
        
        output_dir1 = pathlib.Path(output_dir14).joinpath(f"bat_{ba}_pv_{pv}")
        if not os.path.isdir(output_dir1):
            os.mkdir(output_dir1)
            
        title = f"init_LMP_dispatch_{dispatch}_bat_{batSize}_pv_{pvSize}"
        print(title)

        ####################################
        # First thing: compute the initial Dispatch
        ####################################
        demandProfile, LMP, OperationCost, mOperationCost = SLP_LP_scheduling(batSize, pvSize, output_dir1, vmin, vmax, outDSS=outDSS, plot=plot, freq="h", dispatchType=dispatch)

        # save initial LMP
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        LMP.T.plot(legend=False)
        ax.set_title(title)
        fig.tight_layout()
        output_img = os.path.join(DIR, title + ext)
        plt.savefig(output_img)
        plt.close('all')
        
        # LMP = LMP.iloc[3:,:] # remove first 3 rows
                
        # # Set random seed so results are repeatable
        # np.random.seed(2022) 
        # # define init energy 
        # LMP_size = np.size(LMP,0)
        # initEnergy_list = [np.random.uniform(18, 70) for i in range(LMP_size)]
        # # define ev capacity
        # evCapacity_list = [np.random.uniform(80.5, 118) for i in range(LMP_size)]
        # # define arrival time
        # arrivalTime_list = [f"{np.random.randint(16, 22)}:{np.random.randint(0,2)*3}0" for i in range(LMP_size)]
        # # define departure time
        # departureTime_list = [f"{np.random.randint(6, 12)}:{np.random.randint(0,2)*3}0" for i in range(LMP_size)]
        # # create weights using dirichlet distribution: the sumation add up to 1
        # initW_list = [np.random.dirichlet(np.ones(4),size=1) for i in range(LMP_size)]
        # # LMP index random
        # # LMP_index = np.random.choice(LMP_size, LMP_size, replace=False)
        # LMP_index = LMP.sample(frac=1, random_state=np.random.RandomState(2022)).index 

        # # create smart charging driver object
        # char_obj = smartCharging_driver(ext, arrivalTime_list, departureTime_list, initEnergy_list, evCapacity_list, initW_list) 
        
        # # define folder to store smartcharging results
        # output_dir = pathlib.Path(output_dir1).joinpath("SmartCharging")
        # if not os.path.isdir(output_dir):
        #     os.mkdir(output_dir)
            
        # ####################################
        # # Second thing: compute the decentralized smart charging
        # ####################################
        
        # evh = [-1]
        # evh_size = len(evh)
        # # number of EV loop
        # for ev in range(evh_size): 
        #     # initialize store variables as lists
        #     OpCost_list = list()
        #     mOpCost_list = list()
        #     LMP_list = list()
        #     demand_list = list()

        #     # initial append
        #     LMP_list.append(LMP)
        #     demand_list.append(demandProfile)
        
        #     # create a folder to store LMP per EV
        #     output_dirEV = pathlib.Path(output_dir).joinpath(f"EV_{ev}")
        #     if not os.path.isdir(output_dirEV):
        #         os.mkdir(output_dirEV)
                
        #     # per node LMP comparison
        #     dLMP_list = list()

        #     # prelocate demand difference variable
        #     diffDemand = np.zeros(demandProfile.shape)

        #     # define number of max iterations
        #     maxIter = 30 
        #     sum_dLMP = 100
        #     sum_dLMP_list = list()
        #     tol = 70
        #     it=0
            
        #     # # iterations loop
        #     # while sum_dLMP > tol and it < maxIter: 
                
        #     # novelty criterion
        #     if it == 0:
        #         mean_LMP = LMP_list[-1]
        #     else:
        #         aux2 = [(dLMP/sum(dLMP_list)) for dLMP in  dLMP_list]  #debug             
        #         aux1 = [np.expand_dims((dLMP/sum(dLMP_list)),axis=1) * LMPi.values  for dLMP, LMPi in zip(dLMP_list, LMP_list[1:])]
        #         mean_LMP = pd.DataFrame(sum(aux1), index=LMP.index, columns=LMP.columns)
    
        #     # compute EV corrected demand
        #     newDemand = char_obj.charging_driver(output_dirEV, it, demandProfile, mean_LMP, LMP_index[:evh[ev]], plot)
        #     demand_list.append(newDemand)
            
        #     # compute scheduling with new demand
        #     _, LMP_EVC, OperationCost_EVC, mOperationCost_EVC = SLP_LP_scheduling(batSize, pvSize, output_dirEV, vmin, vmax, userDemand=newDemand, plot=plot, freq="30min", dispatchType=dispatch)

        #     # save EVC LMP
        #     plt.clf()
        #     fig, ax = plt.subplots(figsize=(h,w))
        #     LMP_EVC.T.plot(legend=False)
        #     ax.set_title(f"EVC_LMP - bat:{batSize}_pv:{pvSize}")
        #     fig.tight_layout()
        #     output_img = pathlib.Path(script_path).joinpath(f"EVC_LMP_bat_{batSize}_pv_{pvSize}"+ ext)
        #     plt.savefig(output_img)
        #     plt.close('all')
        
            #     # store corrected values 
            #     OpCost_list.append(OperationCost_EVC) 
            #     mOpCost_list.append(mOperationCost_EVC) 
            #     LMP_list.append(LMP_EVC) 

            #     # store node-base LMP difference
            #     dLMP_list.append(np.linalg.norm(LMP_list[it+1].values - LMP_list[it].values, ord=metric, axis=1))
            #     sum_dLMP = np.linalg.norm(LMP_list[it+1].values - LMP_list[it].values, ord=metric)
            #     sum_dLMP_list.append(sum_dLMP)

            #     # keep track
            #     print(f"bat:{ba}_pv:{pv}_EV:{ev}_it:{it}_diff={sum_dLMP}_cost:{np.round(OpCost_list[-1],2)}")

            #     # Store demand difference
            #     # diffDemand = newDemand - prevDemand
            #     diffDemand = demand_list[it+1].values - demand_list[it].values 

            #     if plot:
            #         diffDemand_pd = pd.DataFrame(diffDemand)
            #         fig, ax = plt.subplots(figsize=(h,w))
            #         sns.heatmap(diffDemand, annot=False)
            #         fig.tight_layout()

            #         output_dirDemand = pathlib.Path(output_dirEV).joinpath("Demand")
            #         if not os.path.isdir(output_dirDemand):
            #             os.mkdir(output_dirDemand)

            #         output_img = pathlib.Path(output_dirDemand).joinpath(f"Demand_diff_it_{it}"+ ext)
            #         plt.savefig(output_img)
            #         plt.close('all')

            #         output_pkl = pathlib.Path(output_dirDemand).joinpath(f"Demand_diff_it_{it}.pkl")
            #         diffDemand_pd.to_pickle(output_pkl)
                
            #     if it != 0:
            #         if mOpCost_list[it] > 1.1*mOpCost_list[it-1]:
            #             break
            #     it += 1
                
            # ##########
            # # node based LMP difference (L2)
            # pd_dLMP = pd.DataFrame(np.stack(dLMP_list,1))
            # dLMP_file = pathlib.Path(output_dirEV).joinpath(f"dLMP_it{it}.pkl")
            # pd_dLMP.to_pickle(dLMP_file)
            # # plot
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(h,w))
            # sns.heatmap(pd_dLMP, annot=False)
            # fig.tight_layout()
            # output_img = pathlib.Path(output_dirEV).joinpath(f"dLMP__it{it}"+ ext)
            # plt.savefig(output_img)
            # plt.close('all')
                    
            # ########################
            # ### metric diff LMP  
            # pd_sum_dLMP = pd.DataFrame(sum_dLMP_list)
            # sum_dLMP_file = pathlib.Path(output_dirEV).joinpath(f"sum_dLMP_it{it}.pkl")
            # pd_sum_dLMP.to_pickle(sum_dLMP_file)
            # # plot
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(h,w))
            # plt.ylim(0.99*min(sum_dLMP_list), 1.01*max(sum_dLMP_list))
            # plt.stem(sum_dLMP_list)
            # ax.set_title(f'{min(sum_dLMP_list)}')
            # fig.tight_layout()
            # output_img = pathlib.Path(output_dirEV).joinpath(f"sum_dLMP_list_it{it}"+ ext)
            # plt.savefig(output_img)
            # plt.close('all')

            # ########################
            # ###  mcosts
            # pd_J = pd.DataFrame(mOpCost_list)
            # J_file = pathlib.Path(output_dirEV).joinpath(f"Jlist_{it}.pkl")
            # pd_J.to_pickle(J_file)
            # # plot
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(h,w))
            # plt.ylim(0.999*min(mOpCost_list), 1.001*max(mOpCost_list))
            # plt.stem(mOpCost_list)
            # ax.set_title(f'{min(mOpCost_list)}')
            # fig.tight_layout()
            # output_img = pathlib.Path(output_dirEV).joinpath(f"mOperation_cost_list_it{it}"+ ext)
            # plt.savefig(output_img)
            # plt.close('all')
