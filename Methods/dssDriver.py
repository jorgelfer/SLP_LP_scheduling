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
    nodeP = {node: 0 for node in nodeNames}
    nodeQ = {node: 0 for node in nodeNames}
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
            P = dss.cktelement_powers()[0::2]
            Q = dss.cktelement_powers()[1::2]
            
            for n, node in enumerate(nodes):
                nodeP[bus + f".{node}"] = abs(P[n])
                nodeQ[bus + f".{node}"] = abs(Q[n])
                
    # powers as array:
    Pa = np.array([nodeP[node] for node in nodeNames])
    Qa = np.array([nodeQ[node] for node in nodeNames])

    return Pa, Qa

#driver function:
def dssDriver(output_dir, iterName, scriptPath, case, dss, dssFile, loadNames, dfDemand, dfDemandQ, dispatchType, vmin, vmax, out=None, plot=True):

    dss.text(f"Compile [{dssFile}]") 
    set_baseline(dss)

    # create a sensitivity object
    sen_obj_0 = sensitivityPy(dss, time=0)
    
    # get all node-based base volts 
    nodeBaseVoltage = sen_obj_0.get_nodeBaseVolts()

    # get all node-based buses, 
    nodeNames = dss.circuit_all_node_names()

    # get all node-based lines names
    nodeLineNames = sen_obj_0.get_nodeLineNames()

    # points in time
    pointsInTime = len(dfDemand.columns)
    
    # prelocation
    v = np.zeros([len(nodeNames), pointsInTime]) # initial voltages
    p = np.zeros([len(nodeNames), pointsInTime]) # initial voltages
    q = np.zeros([len(nodeNames), pointsInTime]) # initial voltages
    pjk = np.zeros([len(nodeLineNames), pointsInTime]) # initial voltages

    # prelocate 
    dvdp_list = list()
    dvdq_list = list()

    # main loop for each load mult
    for t, time in enumerate(dfDemand.columns):
        # fresh compilation to remove previous modifications
        dss.text(f"Compile [{dssFile}]") 
        set_baseline(dss)
        
        #create a sensitivity object
        sen_obj = sensitivityPy(dss, time=time)
        
        # set all loads
        sen_obj.setLoads(dfDemand.loc[:,time], dfDemandQ.loc[:,time], loadNames)
        
        if out is not None:
            sen_obj.modifyDSS(out, nodeBaseVoltage)
        
        dss.text("solve")
        
        # extract power
        p[:,t], q[:,t] = get_nodePowers(dss, nodeNames)

        # extract voltages
        v[:,t] = sen_obj.voltageProfile()
        
        # compute sensitivity
        # dvdp, dvdq = computeSensitivity(scriptPath, case, dss, plot=False, ite=t)
        # dvdp_list.append(dvdp)
        # dvdq_list.append(dvdq)
        
        # extract line flows 
        pjk[:,t] = sen_obj.pjkFlows(nodeLineNames)

    # define outputs
    # dvdp_c = pd.concat(dvdp_list, axis=0)    
    # dvdq_c = pd.concat(dvdq_list, axis=0)
    
    
    dfV = pd.DataFrame(v, index=np.asarray(nodeNames), columns=dfDemand.columns)
    dfP = pd.DataFrame(p, np.asarray(nodeNames), columns=dfDemand.columns)
    dfQ = pd.DataFrame(q, np.asarray(nodeNames), columns=dfDemand.columns)
    dfPjks = pd.DataFrame(pjk, index=np.asarray(nodeLineNames), columns=dfDemand.columns)

    if plot:
        #plot results
        plot_obj = plottingDispatch(output_dir, iterName, pointsInTime, scriptPath, vmin, vmax, dispatchType=dispatchType)
        
        #plot Line Limits\
        Lmax, Linfo = load_lineLimits(scriptPath, case, PointsInTime=pointsInTime) 
        plot_obj.plot_Pjk(dfPjks, Linfo, Lmax)
    
        #plot voltage constraints 
        plot_obj.plot_voltage(nodeBaseVoltage, dfV, dfDemand.any(axis=1))

    return dfP, dfV, dfPjks, nodeBaseVoltage


