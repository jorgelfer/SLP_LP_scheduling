import numpy as np
import py_dss_interface
import os
import pathlib
import sympy
import pandas as pd
import cmath

class reactiveCorrection:

    def __init__(self, dss):
        self.dss = dss

    def compute_correction(self, vo, baseVoltage, lineNames):
        
        vo = vo / 1000
        
        # prelocation
        Pjk_lim = np.zeros((len(lineNames), len(vo.columns)))

        # get line Yprim
        yprim_phase, buses, Sjk_max = self.__getLineYprim(baseVoltage)

        for t, time in enumerate(vo.columns): 
            for l, line in enumerate(lineNames):
                # assing yprim components
                gjk = yprim_phase[line][0]
                bjk = yprim_phase[line][1]
                # assign sending and receiving bus voltages
                vj = vo.loc[buses[line][0], time]
                vk = vo.loc[buses[line][1], time]
                # compute constants
                Pjk_C = (vj**2) * gjk
                Qjk_C = -(vj**2) * bjk
                Sjk_C = vj * vk * np.sqrt(gjk**2 + bjk**2) 
                # MC = -Pjk_C**2 - Qjk_C**2 + Sjk_C**2

                # # solve system of equation
                # a = Pjk_C**2 + Qjk_C**2
                # b = -Pjk_C*((Sjk_max[line])**2 - MC)
                # c = 0.25*((Sjk_max[line])**2 - MC)**2 - (Qjk_C)**2 * (Sjk_max[line])**2
                MC = (Sjk_max[line]**2) - (Sjk_C**2) + (Pjk_C**2) + (Qjk_C**2) 
                a = 4*(Pjk_C**2 + Qjk_C**2)
                b = - 2 * Pjk_C * MC
                c = (MC)**2 - 4*(Qjk_C**2)*(Sjk_C**2)
                # x, y = sympy.symbols("x y", real=True) 
                # eq1 = sympy.Eq((x - Pjk_C)**2 + (y - Qjk_C)**2, Sjk_C**2)
                # eq2 = sympy.Eq(x**2 + y**2, Sjk_max[line]**2)

                # sympy.solve([eq1, eq2])
                
                d = (b**2) - (4*a*c)
                Pjk_lim = (-b + np.sqrt(d))/(2*a)

        Pjk_lim = pd.DataFrame(Pjk_lim, index=lineNames, columns=vo.columns)
        return Pjk_lim

    # Helper methods
    def __getLineYprim(self, baseVoltage):

        # prelocate 
        elements = self.dss.circuit_all_element_names()
        lname_phase = list()
        yprim_phase = dict()
        buses_dict = dict()
        Sjk_max = dict()
    
        for i, elem in enumerate(elements):
            self.dss.circuit_set_active_element(elem)
            if "Line" in elem:
                # get node-based line names
                buses = self.dss.cktelement_read_bus_names()
                
                # get number of nodes including reference
                n = len(self.dss.cktelement_node_order())
                
                # extract and organize yprim
                yprim = self.dss.cktelement_y_prim()
                Yprim_tmp = np.asarray(yprim).reshape((2*n,n), order="F")
                Yprim =Yprim_tmp.T
                Yprim = Yprim[:int(n/2),:n]
                Gprim = np.diag(Yprim[:, 0::2])
                Bprim = np.diag(Yprim[:, 1::2])
                
                # get nodes and discard the reference
                nodes = [i for i in self.dss.cktelement_node_order() if i != 0]
                
                # reorder the number of nodes
                nodes = np.asarray(nodes).reshape((int(len(nodes)/2),-1),order="F")                
                cont = 0
                for t1n, t2n in zip(nodes[:,0],nodes[:,1]):
                    # store line name per phase
                    lname_phase.append("L"+ buses[0].split(".")[0] + f".{t1n}" + "-" + buses[1].split(".")[0] + f".{t2n}")
                    # store line Yprim elements
                    yprim_phase[lname_phase[-1]] = [Gprim[cont], Bprim[cont]]
                    # store sending and receiving buses
                    buses_dict[lname_phase[-1]] = [buses[0].split(".")[0] + f".{t1n}", buses[1].split(".")[0] + f".{t2n}"]
                    # compute and store max rated power
                    Sjk_max[lname_phase[-1]] = self.dss.cktelement_read_norm_amps() * baseVoltage[buses[0].split(".")[0] + f".{t1n}"]
                    # increase cont
                    cont += 1

            elif "Transformer" in elem:
                # get node-based line names
                buses = self.dss.cktelement_read_bus_names()
                
                # get number of nodes including reference
                n = len(self.dss.cktelement_node_order())
                
                # extract and organize yprim
                yprim = self.dss.cktelement_y_prim()
                Yprim_tmp = np.asarray(yprim).reshape((2*n,n), order="F")
                Yprim =Yprim_tmp.T
                Yprim = Yprim[:int(n/2),:n]
                Gprim = np.diag(Yprim[:, 0::2])
                Bprim = np.diag(Yprim[:, 1::2])
                
                # get nodes and discard the reference
                nodes = [i for i in self.dss.cktelement_node_order() if i != 0]
                
                # reorder the number of nodes
                nodes = np.asarray(nodes).reshape((int(len(nodes)/2),-1),order="F")                
                cont = 0
                for t1n, t2n in zip(nodes[:,0],nodes[:,1]):
                    # store line name per phase
                    lname_phase.append("T"+ buses[0].split(".")[0] + f".{t1n}" + "-" + buses[1].split(".")[0] + f".{t2n}")
                    # store line Yprim elements
                    yprim_phase[lname_phase[-1]] = [Gprim[cont], Bprim[cont]]
                    # store sending and receiving buses
                    buses_dict[lname_phase[-1]] = [buses[0].split(".")[0] + f".{t1n}", buses[1].split(".")[0] + f".{t2n}"]
                    # compute and store max rated power
                    Sjk_max[lname_phase[-1]] = self.dss.cktelement_read_norm_amps() * baseVoltage[buses[0].split(".")[0] + f".{t1n}"]
                    # increase cont
                    cont += 1

        return yprim_phase, buses_dict, Sjk_max 


#+===========================================================================
def main():

    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("..", "EV_data", "123bus", "IEEE123Master.dss")
    
    dss = py_dss_interface.DSSDLL()
    dss.text(f"Compile [{dss_file}]")

    # compute yprimitive
    Qobj = reactiveCorrection(dss) 
    lines, ymat = Qobj.getLineYprim()

#+===========================================================================
if __name__ == "__main__":
    main()

