# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:00:01 2022

@author: tefav
"""

from Methods.sensitivityPy import sensitivityPy 
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt


def disableGen(dss, gen):
    dss.circuit_set_active_element('generator.' + gen)
    #debug
    # a = dss.cktelement_read_enabled() == 1
    dss.text(f"disable generator.{gen}")

def computeSensitivity(script_path, case, dss, plot, ite):

    # create a sensitivity object
    sen_obj = sensitivityPy(dss, time=0)
    
    # get all node-based base volts 
    nodeBaseVoltage = sen_obj.get_nodeBaseVolts()

    # get all node-based buses, 
    nodeNames = dss.circuit_all_node_names()

    # get all node-based lines names
    nodeLineNames = sen_obj.get_nodeLineNames()

    # get base voltage
    baseVolts = sen_obj.voltageProfile()
    
    # get base pjk 
    basePjk = sen_obj.pjkFlows(nodeLineNames)
    
    # prelocate to store the sensitivity matrices
    PTDF_jk = np.zeros([len(nodeLineNames),len(nodeNames)]) # containing flows Pjk
    VS = np.zeros([len(nodeNames),len(nodeNames)]) # containing volts 
    
    # main loop through all nodes 
    for n, node in enumerate(nodeNames):

        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)

        # Perturb DSS with small gen 
        sen_obj.perturbDSS(node, kv=nodeBaseVoltage[node], kw=10) # 10 kw
        
        dss.text("solve")

        # compute Voltage sensitivity
        currVolts = sen_obj.voltageProfile()
        VS[:,n] =  currVolts- baseVolts
        
        # compute PTDF
        currPjk = sen_obj.pjkFlows(nodeLineNames)
        PTDF_jk[:,n] =  currPjk - basePjk
        
        # disable Generator
        name = node.replace(".","_")
        disableGen(dss, name)
        
    # save
    dfVS = pd.DataFrame(VS, np.asarray(nodeNames), np.asarray(nodeNames))
    dfPjk = pd.DataFrame(PTDF_jk,np.asarray(nodeLineNames), np.asarray(nodeNames))
    

    if plot:
        h = 20
        w = 20
        ext = '.png'
        
        # VoltageSensitivity
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))                
        ax = sns.heatmap(dfVS, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", f"VoltageSensitivity_{ite}" + ext)
        plt.savefig(output_img)
        plt.close('all')
        
        # PTDF
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))                
        ax = sns.heatmap(dfPjk, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs",f"PTDF_{ite}" + ext)
        plt.savefig(output_img)
        plt.close('all')
        
    return dfVS, dfPjk
