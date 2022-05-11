# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:22:51 2021

@author: tefav
"""
# required for processing
import pathlib
import os

import py_dss_interface
import numpy as np
from Methods.dssDriver import dssDriver
from Methods.schedulingDriver import schedulingDriver
from Methods.initDemandProfile import getInitDemand 
from Methods.computeSensitivity import computeSensitivity
from Methods.reactiveCorrection import reactiveCorrection 

#required for plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

# the save methods are only for testing
def save_initDSS(script_path, Pg_0, v_0, Pjk_0, v_base, loadNames, initDemand):
    # save initial DSS values
    outDSS = dict()
    # initVolts.to_pickle(pathlib.Path(script_path).joinpath("share","initVolts.pkl"))
    outDSS['initPower'] = Pg_0
    # initVolts.to_pickle(pathlib.Path(script_path).joinpath("share","initVolts.pkl"))
    outDSS['initVolts'] = v_0
    # initPjks.to_pickle(pathlib.Path(script_path).joinpath("share","initPjks.pkl"))
    outDSS['initPjks'] = Pjk_0
    # nodeBaseVolts.to_pickle(pathlib.Path(script_path).joinpath("share","nodeBaseVolts.pkl"))
    outDSS['nodeBaseVolts'] = v_base 
    # loadNames.to_pickle(pathlib.Path(script_path).joinpath("share","loadNames.pkl"))
    outDSS['loadNames'] = loadNames
    # initDemand.to_pickle(pathlib.Path(script_path).joinpath("share","initDemand.pkl"))
    outDSS['initDemand'] = initDemand

    return outDSS

def save_ES(script_path, outGen, outDR, outPsc, outPsd):
    # save optimization values
    outES = dict()
    # outGen.to_pickle(pathlib.Path(script_path).joinpath("share","outGen.pkl"))
    outES['Gen'] = outGen
    # outDR.to_pickle(pathlib.Path(script_path).joinpath("share","outDR.pkl"))
    outES['DR'] = outDR
    # outPsc.to_pickle(pathlib.Path(script_path).joinpath("share","outPchar.pkl"))
    outES['Pchar'] = outPsc
    # outPsd.to_pickle(pathlib.Path(script_path).joinpath("share","outPdis.pkl"))
    outES['Pdis'] = outPsd
    return outES

def SLP_LP_scheduling(output_dir, userDemand=None, plot=False, freq="15min", dispatchType='SLP'):

    # execute the DSS model
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("EV_data", "123bus", "IEEE123Master.dss")
    
    dss = py_dss_interface.DSSDLL()
    dss.text(f"Compile [{dss_file}]")
    
    # initialization
    case = '123bus'
    
    #compute sensitivities for the test case
    compute = False
    if compute:
        computeSensitivity(script_path, case, dss, dss_file, plot)
    
    # get init load
    loadNames, dfDemand = getInitDemand(script_path, dss, freq)
    
    # correct native load by user demand
    if userDemand is not None:
        dfDemand.loc[loadNames.index,:] = userDemand
        
    #Dss driver function
    Pg_0, v_0, Pjk_0, v_base = dssDriver(output_dir, 'InitDSS', script_path, case, dss, dss_file, loadNames, dfDemand, dispatchType, out=None, plot=plot)
    outDSS = save_initDSS(script_path, Pg_0, v_0, Pjk_0, v_base, loadNames, dfDemand)

    # # reactive power correction
    # Q_obj = reactiveCorrection(dss) 
    # Pjk_lim = Q_obj.compute_correction(v_0, v_base, Pjk_0.index)
    
    #Energy scheduling driver function   
    outGen, outDR, outPchar, outPdis, outLMP, costPdr, cgn, mobj = schedulingDriver(output_dir, 'Dispatch', freq, script_path, case, outDSS, dispatchType, plot=plot)
    outES = save_ES(script_path, outGen, outDR, outPchar, outPdis)
    
    # normalization 
    loadNames = outDSS['loadNames']
    dfDemand = outDSS['initDemand']
    
    #corrected dss driver function
    Pg, v, Pjk, v_base = dssDriver(output_dir, 'DispatchDSS', script_path, case, dss, dss_file, loadNames, dfDemand, dispatchType, out=outES, plot=plot)

    # initial power 
    Pg = np.reshape(Pg.values.T, np.size(Pg), order="F")
    costPg = cgn @ Pg

    operationCost = costPdr[0] + costPg[0]
        
    return dfDemand.loc[loadNames.index,:], outLMP, operationCost, mobj
        
# demandProfile, LMP = LP_scheduling(userDemand=None, plot=True)           
        
