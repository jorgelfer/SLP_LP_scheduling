"""
# -*- coding: utf-8 -*-
# @Time    : 10/11/2021 6:09 PM
# @Author  : Jorge Fernandez
"""

from Methods.sensitivityPy import sensitivityPy 
import numpy as np
import pandas as pd
import pathlib
from Methods.plotting import plottingDispatch
from Methods.computeSensitivityHourly import computeSensitivity
# import numba
# from numba import jit
# @jit(nopython=True, parallel=True)

def set_baseline(dss):
    
    dss.text("Set Maxiterations=100")
    dss.text("Set controlmode=Off") # this is for not using the default regulator
    
def load_lineLimits(script_path, case, PointsInTime, PTDF=None, DR=True):

    if DR == True:
        Lmax_file = pathlib.Path(script_path).joinpath("inputs", case, "Pjk_ratings.pkl") #high limits: Pjk_ratings_Nov-11-2021_1745 ## low limits:Pjk_ratings_Nov-20-2021_1154
        Lmaxi = pd.read_pickle(Lmax_file)[0]
    else:
        #debug:
        Lmaxi = 2000 * np.ones((len(PTDF),1))

    Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
    Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")
    # Line Info
    Linfo_file = pathlib.Path(script_path).joinpath("inputs", case, "LineInfo.pkl")
    Linfo = pd.read_pickle(Linfo_file)

    return Lmax, Linfo

def get_nodePowers(dss, nodeNames):
    # initialize power dictionary
    nodePower = {node: 0 for node in nodeNames}
    elements = dss.circuit_all_element_names()

    for i, elem in enumerate(elements):
        dss.circuit_set_active_element(elem)
        if "Vsource" in elem:
            # get node-based line names
            buses = dss.cktelement_read_bus_names()
            bus = buses[0]
            
            # get this element node and discard the reference
            nodes = [i for i in dss.cktelement_node_order() if i != 0]
            
            # reorder the number of nodes
            power = dss.cktelement_powers()[0::2]
            
            for n, node in enumerate(nodes):
                nodePower[bus + f".{node}"] = abs(power[n])

    return np.array([nodePower[node] for node in nodeNames])

#driver function:
def dssDriver(output_dir, iterName, scriptPath, case, dss, dssFile, loadNames, dfDemand, dispatchType, out=None, plot=True):

    set_baseline(dss)

    # create a sensitivity object
    sen_obj = sensitivityPy(dss, time=0)
    
    # get all node-based base volts 
    nodeBaseVoltage = sen_obj.get_nodeBaseVolts()

    # get all node-based buses, 
    nodeNames = dss.circuit_all_node_names()

    # get all node-based lines names
    nodeLineNames = sen_obj.get_nodeLineNames()

    # points in time
    pointsInTime = len(dfDemand.columns)
    
    # prelocation
    volts = np.zeros([len(nodeNames), pointsInTime]) # initial voltages
    powers = np.zeros([len(nodeNames), pointsInTime]) # initial voltages
    pjk= np.zeros([len(nodeLineNames), pointsInTime]) # initial voltages
    
    # main loop for each load mult
    for t, time in enumerate(dfDemand.columns):
        # fresh compilation to remove previous modifications
        dss.text(f"Compile [{dssFile}]") 
        # set_baseline(dss)
        
        #create a sensitivity object
        sen_obj = sensitivityPy(dss, time=time)
        
        # set all loads
        sen_obj.setLoads(dfDemand.loc[:,time], loadNames)
        
        if out is not None:
            sen_obj.modifyDSS(out, nodeBaseVoltage)
        
        dss.text("solve")
        
        # extract power
        powers[:,t] = get_nodePowers(dss, nodeNames)
        
        # extract voltages
        volts[:,t] = sen_obj.voltageProfile()

        # extract line flows 
        pjk[:,t] = sen_obj.pjkFlows(nodeLineNames)
    
    # create dataframes
    dfPow = pd.DataFrame(powers, index=np.asarray(nodeNames), columns=dfDemand.columns)
    dfVolts = pd.DataFrame(volts, index=np.asarray(nodeNames), columns=dfDemand.columns)
    dfPjks = pd.DataFrame(pjk, index=np.asarray(nodeLineNames), columns=dfDemand.columns)

    if plot:
        #plot results
        plot_obj = plottingDispatch(output_dir, iterName, PointsInTime=pointsInTime, script_path=scriptPath, dispatchType=dispatchType)
        
        #plot Line Limits\
        Lmax, Linfo = load_lineLimits(scriptPath, case, PointsInTime=pointsInTime) 
        plot_obj.plot_Pjk(dfPjks, Linfo, Lmax)
    
        #plot voltage constraints 
        plot_obj.plot_voltage(nodeBaseVoltage, dfVolts, dfDemand.any(axis=1))

    return dfPow, dfVolts, dfPjks, nodeBaseVoltage


