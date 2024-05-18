"""
by Jorge
"""

# required for processing
import numpy as np
import pandas as pd
import pathlib
#required for plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from Methods.SLP_dispatch import SLP_dispatch
from Methods.LP_dispatch import LP_dispatch
from Methods.plotting import plottingDispatch
from Methods.loadHelper import loadHelper

# Methods
def create_output(Pg, Pdr, Pchar, Pdis, LMP, load, PTDF, lnodes, Snodes, PVnodes, timeVec, PV, storage):
    """function to create outputs"""
    outLMP = pd.DataFrame(LMP[lnodes,:], np.asarray(PTDF.columns[lnodes]), timeVec) 
    lnames = [load.loc[n] for n in PTDF.columns[lnodes]]
    outDR = pd.DataFrame(Pdr[lnodes,:], np.asarray(lnames), timeVec)
    
    # check PV
    if PV:
        outGen = pd.DataFrame(Pg[PVnodes,:], np.asarray(PTDF.columns[PVnodes]), timeVec)
    else:
        outGen = None
    # check storage
    if storage:
        outPchar = pd.DataFrame(Pchar, np.asarray(PTDF.columns[Snodes]), timeVec) 
        outPdis = pd.DataFrame(Pdis, np.asarray(PTDF.columns[Snodes]), timeVec) 
    else:
        outPchar = None
        outPdis = None

    return outGen, outDR, outPchar, outPdis, outLMP

def create_battery(PTDF, pointsInTime, sbus='83', batSize=300):
    """function to define battery parameters"""

    batt = dict()
    numBatteries= 3
    batt['numBatteries'] = numBatteries
    BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
    BatIncidence[PTDF.columns == sbus +'.1', 0] = 1
    BatIncidence[PTDF.columns == sbus +'.2', 1] = 1
    BatIncidence[PTDF.columns == sbus +'.3', 2] = 1
    batt['BatIncidence'] = BatIncidence
    
    BatSizes = batSize * np.ones((1,numBatteries))
    batt['BatSizes'] = BatSizes
    
    BatChargingLimits = (24/pointsInTime)*100*np.ones((1,numBatteries))
    batt['BatChargingLimits'] = BatChargingLimits
    BatEfficiencies = 0.97*np.ones((1,numBatteries))
    batt['BatEfficiencies'] = BatEfficiencies
    
    np.random.seed(2022) # Set random seed so results are repeatable
    BatInitEnergy = BatSizes * np.random.uniform(0.5, 0.8, size=(1,numBatteries)) 
    
    batt['BatInitEnergy'] = BatInitEnergy
    Pbatcost = 0.0
    batt['Pbatcost'] = Pbatcost
    ccharbat = Pbatcost * np.ones((1,2*numBatteries*pointsInTime))
    batt['ccharbat'] = ccharbat
    ccapacity = Pbatcost * np.ones((1, numBatteries*(pointsInTime + 1)))
    batt['ccapacity'] = ccapacity
    batt['BatPenalty'] = np.ones((1,numBatteries)) 

    return batt

def load_generationCosts(script_path, n, pointsInTime, freq):
    '''function to load generations costs and perform interpolation'''
    
    GenPrice_file = pathlib.Path(script_path).joinpath("inputs", "HourlyMarginalPrice.xlsx")
    tcost = pd.read_excel(GenPrice_file)
    gCost = 10000*np.ones((n,pointsInTime))
    
    # create load helper method
    help_obj = loadHelper(initfreq = 'h', finalFreq = freq, price=True)
    
    cost_wednesday = pd.Series(tcost.values[225,1:-1]) # 2018-08-14
    # call method for processing series
    cost_wednesday = help_obj.process_pdSeries(cost_wednesday)
    cost_wednesday = np.squeeze(cost_wednesday.values * 1e-3) # $/MWh -> $/kWh

    gCost[0,:] = cost_wednesday 
    gCost[1,:] = cost_wednesday
    gCost[2,:] = cost_wednesday
    
    return gCost, cost_wednesday

def create_PVsystems(freq, Gmax, PTDF, gCost, cost_wednesday, pointsInTime, pv1bus='634', pv2bus='680', pvSize=100):
    '''function to Define the utility scale PVs'''    

    nodesPV1 = [pv1bus +'.1',pv1bus +'.2',pv1bus +'.3']
    nodesPV2 = [pv2bus +'.1',pv2bus +'.2',pv2bus +'.3']

    # define the PV location
    PV1 = np.stack([PTDF.columns == nodesPV1[0], PTDF.columns ==nodesPV1[1], PTDF.columns ==nodesPV1[2]], axis=1)
    PV2 = np.stack([PTDF.columns ==nodesPV2[0],PTDF.columns ==nodesPV2[1],PTDF.columns ==nodesPV2[2]], axis=1)
    PVnodes = np.any(np.concatenate([PV1,PV2],axis=1), axis=1)
    
    # define the maximum output
    Gmax[np.where(np.any(PV1,axis=1))[0]] =  pvSize        #% Utility scale Solar PV    
    Gmax[np.where(np.any(PV2,axis=1))[0]] =  pvSize        #% Utility scale Solar PV

    if pv1bus == pv2bus:
        Gmax[np.where(np.any(PV1,axis=1))[0]] =  300        #% Utility scale Solar PV
    
    # define the cost
    gCost[PVnodes] = 0.1*cost_wednesday#*np.mean(cost_wednesday)
    
    # create load helper method
    help_obj = loadHelper(initfreq = 'h', finalFreq = freq)
    
    # Estimate a PV Profile
    np.random.seed(2022)
    a = np.sin(np.linspace(-4,19,24)*np.pi/15) - 0.5 + np.random.rand(24)*0.2
    a[a<0] = 0
    a = a/max(a)
    
    # call method for processing series
    PVProfile = help_obj.process_pdSeries(pd.Series(a))
    PVProfile[PVProfile<0.01] = 0
    PVProfile = np.squeeze(PVProfile.values)

    return Gmax, gCost, PVnodes, PVProfile

def compute_penaltyFactors(batt, PTDF): 
    '''function to Compute penalty factors'''
        
    # compute dPgref
    dPgref = np.min(PTDF[:3], axis=0)
    
    # dPl/dPgi = 1 - (- dPgref/dPgi) -> eq. L9_25
    
    # ITLi = dPL/dPGi 
    ITL = 1 + dPgref # Considering a PTDF with transfer from bus i to the slack. If PTDF is calculated in the converse, then it will be 1 - dPgref
    
    Pf = 1 / (1- ITL)
    
    # substation correction
    Pf['150.1'] = 1
    Pf['150.2'] = 1
    Pf['150.3'] = 1
    # batt incidence
    BatIncidence = batt['BatIncidence'] 
    
    # nodes with batt 
    nodes = np.where(np.any(BatIncidence,1))[0]
    
    # assign penalty factors 
    batt['BatPenalty'] = np.asarray([Pf.values[n] for n in nodes])#min([Pf.values[n] for n in nodes[0]])*np.ones((1,3)) 

    return batt, Pf, nodes

def schedulingDriver(batSize, pvSize, output_dir, iterName, freq, script_path, case, outDSS, dispatchType, vmin, vmax, PF=False, voltage=True, DR=True, plot=False):
    
    # define Storage and PV
    storage=False

    # extract DSS results
    loadNames = outDSS['loadNames'] 
    Pg_0 = outDSS['initPower']
    v_0 = outDSS['initVolts']
    v_base = outDSS['nodeBaseVolts']
    Pjk_0 = outDSS['initPjks']
    demandProfile = outDSS['initDemand']
    PTDF = outDSS['PTDF']
    Pdr_0 = outDSS['Pdr_0'] 
    pf = outDSS["pf"]
    clin = outDSS['clin']
    Pjk_lim = outDSS['Pjk_lim']
    cdr = outDSS['cdr']
    Gmax = outDSS['Gmax']
    cgn = outDSS['cgn']
    dvdp = outDSS['dvdp']
    batt = outDSS['batt']
    
    # derived values
    PTDF = PTDF.round()
    pointsInTime = v_0.shape[1]
        
    # call the OPF method
    if dispatchType == 'SLP':
        # create an instance of the dispatch class
        dispatch_obj = SLP_dispatch(pf, PTDF, batt, Pjk_lim, Gmax, cgn, clin, cdr, v_base, dvdp, storage, vmin, vmax)
        x, m, LMP = dispatch_obj.PTDF_SLP_OPF(demandProfile, Pjk_0, v_0, Pg_0, Pdr_0)
    else:
        dispatch_obj = LP_dispatch(pf, PTDF, batt, Pjk_lim, Gmax, cgn, clin, cdr, v_base, dvdp, storage, vmin, vmax)
        x, m, LMP, Ain = dispatch_obj.PTDF_LP_OPF(demandProfile, Pjk_0, v_0, Pg_0, Pdr_0)

    print('Obj: %g' % m.objVal) 
    PV=False
    PVnodes = None
    Snodes = np.where(np.any(batt["BatIncidence"],1))[0]

    #Create plot object
    plot_obj = plottingDispatch(output_dir, iterName, pointsInTime, script_path, vmin, vmin, PTDF=PTDF, dispatchType=dispatchType)
    
    # extract dispatch results
    Pg, Pdr, Pij, Pchar, Pdis, E = plot_obj.extractResults(x=x, DR=DR, Storage=storage, batt=batt)

    ################
    #operation costs
    ################
    Pdr_1d = np.reshape(Pdr, np.size(Pdr), order="F")
    costPdr = cdr @ Pdr_1d.T
    
    # extract LMP results
    LMP_Pg, LMP_Pdr, LMP_Pij, LMP_Pchar, LMP_Pdis, LMP_E = plot_obj.extractLMP(LMP, DR, storage, batt)
    demandProfilei = demandProfile.any(axis=1)
    lnodes = np.where(demandProfilei)[0]    
    # Define the output as Pandas DataFrame
    outGen, outDR, outPchar, outPdis, outLMP = create_output(Pg, Pdr, Pchar, Pdis, LMP_Pg, loadNames, PTDF, lnodes=lnodes, Snodes=Snodes, PVnodes=PVnodes, timeVec=v_0.columns, PV=PV, storage=storage)
    
    return outGen, outDR, outPchar, outPdis, outLMP, costPdr, cgn, m.objVal
