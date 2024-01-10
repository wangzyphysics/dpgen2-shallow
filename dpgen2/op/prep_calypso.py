import json
import pickle
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)

from dpgen2.constants import (
    calypso_input_file,
    calypso_task_pattern,
    calypso_run_opt_file,
    calypso_check_opt_file,
    model_name_pattern,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)


class PrepCalyInput(OP):
    r"""Prepare the working directories and input file for generating structures.

    A calypso input file will be generated and saved in working
    directory (defined by `ip["task"]`), according to the
    given parameters (defined by `ip["caly_input"]`).
    The paths of the directory will be returned as `op["work_path"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "caly_inputs": List[dict],  # calypso input params
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_paths": Artifact(Path),  # task path containing all structures
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `caly_inputs` : (`List[dict]`) Definitions for CALYPSO input file.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation. The first model with be used to drive structure prediction simulation.

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of CALYPSO tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the task containing input files (`input.dat` and `calypso_run_opt.py`) needed to generate structures by CALYPSO and make structure optimization with DP model.
        """

        models = ip["models"]
        model_files = [Path(ii).resolve() for ii in models]

        cc = 0
        task_paths = []
        caly_inputs = ip["caly_inputs"]
        for caly_input in caly_inputs:
            update = caly_input.pop("UpDate", True)
            if update:
                for key in necessary_keys.keys():  # All necessary keys must be included or raise.
                    necessary_keys[key] = caly_input.pop(key)
                default_key_value.update(caly_input)
                default_key_value.update(necessary_keys)
            else:
                default_key_value = caly_input
            tname = _mk_task_from_dict(default_key_value, cc)

            # is model_files exist or mname? why link mname into model_files
            for idx, mm in enumerate(model_files):
                mname = model_name_pattern % (idx)
                Path(mname).symlink_to(mm)
            # -----------------------------------------------------------------

            task_paths.append(tname)
            cc += 1
        task_names = [str(ii) for ii in task_paths]
        return OPIO(
            {
                "task_names": task_names,
                "task_paths": task_paths,
            }
        )

PrepExplorationTaskGroup = PrepCalyInput

def _mk_task_from_dict(mapping, cc):
    tname = Path(calypso_task_pattern % cc)
    tname.mkdir(exist_ok=True, parents=True)

    distanceofion = mapping.pop("DistanceOfIon")
    vsc = mapping.pop("VSC", "F").lower().startswith("t")
    if vsc:
        ctrlrange = mapping.pop("CtrlRange")

    file_str = ""
    for key, value in mapping.items():
        file_str += f"{key} = {str(value)}\n"
    file_str += "@DistanceOfIon\n"
    file_str += distanceofion + "\n"
    file_str += "@End\n"
    if vsc:
        file_str += "@CtrlRange\n"
        file_str += ctrlrange + "\n"
        file_str += "@End\n"
    (tname / calypso_input_file).write_text(file_str)

    fmax = mapping.get("fmax", 0.01)
    pstress = mapping.get("PSTRESS", 0)
    calypso_run_opt_str += f"""
    if __name__ == '__main__':
        run_opt({fmax}, {pstress})"""
    (tname / calypso_run_opt_file).write_text(calypso_run_opt_str)
    (tname / calypso_check_opt_file).write_text(calypso_check_opt_str)
    return tname

necessary_keys = {
    "NumberOfSpecies": "",
    "NameOfAtoms": "",
    "AtomicNumber": "",
    "NumberOfAtoms": "",
    "distanceofion": "",
    # "@DistanceOfIon
    # "@End
}

default_key_value = {
    "SystemName": "CALYPSO",
    "NumberOfFormula": "1 1",
    "PSTRESS": "0",
    "fmax": 0.01,
    "Volume": 0,
    "Ialgo": 2,
    "PsoRatio": 0.6,
    "PopSize": 30,
    "MaxStep": 10,
    "ICode": 15,
    "NumberOfLbest": 4,
    "NumberOfLocalOptim": 3,
    "Command": "sh submit.sh",
    "MaxTime": 9000,
    "GenType": 1,
    "PickUp": "F",
    "PickStep": 3,
    "Parallel": "F",
    "LMC": "F",
    "Split": "T",
    "SpeSpaceGroup": "2 230",
    }

vsc_keys = {
    "VSC": "F",
    "MaxNumAtom": 100,
    "CtrlRange":"",
    # @CtrlRange
    # @end
}

calypso_run_opt_str = """#!/usr/bin/env python3

import os, time
import numpy as np
from ase.io import read 
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from deepmd.calculator import DP 
'''
structure optimization with DP model and ASE
PSTRESS and fmax should exist in input.dat
'''

def Get_Element_Num(elements):
    '''Using the Atoms.symples to Know Element&Num'''
    element = []
    ele = {}
    element.append(elements[0])
    for x in elements: 
        if x not in element :
            element.append(x)
    for x in element: 
        ele[x] = elements.count(x)
    return element, ele 
        
def Write_Contcar(element, ele, lat, pos):
    '''Write CONTCAR''' 
    f = open('CONTCAR','w')
    f.write('ASE-DP-OPT\n')
    f.write('1.0\n') 
    for i in range(3):
        f.write('%15.10f %15.10f %15.10f\n' % tuple(lat[i]))
    for x in element: 
        f.write(x + '  ')
    f.write('\n') 
    for x in element:
        f.write(str(ele[x]) + '  ') 
    f.write('\n') 
    f.write('Direct\n')
    na = sum(ele.values())
    dpos = np.dot(pos,np.linalg.inv(lat))
    for i in range(na): 
        f.write('%15.10f %15.10f %15.10f\n' % tuple(dpos[i]))
        
def Write_Outcar(element, ele, volume, lat, pos, ene, force, stress,pstress):
    '''Write OUTCAR'''
    f = open('OUTCAR','w')
    for x in element: 
        f.write('VRHFIN =' + str(x) + '\n')
    f.write('ions per type =')
    for x in element:
        f.write('%5d' % ele[x])
    f.write('\nDirection     XX             YY             ZZ             XY             YZ             ZX\n')
    f.write('in kB') 
    f.write('%15.6f' % stress[0])
    f.write('%15.6f' % stress[1])
    f.write('%15.6f' % stress[2])
    f.write('%15.6f' % stress[3])
    f.write('%15.6f' % stress[4])
    f.write('%15.6f' % stress[5])
    f.write('\n') 
    ext_pressure = np.sum(stress[0] + stress[1] + stress[2])/3.0 - pstress
    f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\n'% (ext_pressure, pstress))
    f.write('volume of cell : %20.6f\n' % volume)
    f.write('direct lattice vectors\n')
    for i in range(3):
        f.write('%10.6f %10.6f %10.6f\n' % tuple(lat[i]))
    f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\n')
    f.write('-------------------------------------------------------------------\n')
    na = sum(ele.values())
    for i in range(na):
        f.write('%15.6f %15.6f %15.6f' % tuple(pos[i])) 
        f.write('%15.6f %15.6f %15.6f\n' % tuple(force[i]))
    f.write('-------------------------------------------------------------------\n')
    f.write('energy  without entropy= %20.6f %20.6f\n' % (ene, ene/na))
    enthalpy = ene + pstress * volume / 1602.17733      
    f.write('enthalpy is  TOTEN    = %20.6f %20.6f\n' % (enthalpy, enthalpy/na)) 

def run_opt(fmax, stress):
    '''Using the ASE&DP to Optimize Configures'''
    
    calc = DP(model='frozen_model.pb')    # init the model before iteration

    Opt_Step = 1000 
    start = time.time()
    # pstress kbar
    pstress = stress
    # kBar to eV/A^3
    # 1 eV/A^3 = 160.21766028 GPa
    # 1 / 160.21766028 ~ 0.006242
    aim_stress = 1.0 * pstress* 0.01 * 0.6242 / 10.0
    to_be_opti = read('POSCAR')
    to_be_opti.calc = calc
    ucf = UnitCellFilter(to_be_opti, scalar_pressure=aim_stress)
    # opt
    opt = LBFGS(ucf,trajectory='traj.traj')
    opt.run(fmax=fmax,steps=Opt_Step)
    
    atoms_lat = to_be_opti.cell
    atoms_pos = to_be_opti.positions
    atoms_force = to_be_opti.get_forces()
    atoms_stress = to_be_opti.get_stress()
    # eV/A^3 to GPa
    atoms_stress = atoms_stress/(0.01*0.6242)
    atoms_symbols = to_be_opti.get_chemical_symbols()
    atoms_ene = to_be_opti.get_potential_energy()
    atoms_vol = to_be_opti.get_volume()
    element, ele = Get_Element_Num(atoms_symbols)

    Write_Contcar(element, ele, atoms_lat, atoms_pos)
    Write_Outcar(element, ele, atoms_vol, atoms_lat, atoms_pos,atoms_ene, atoms_force, atoms_stress * -10.0, pstress)"""

calypso_check_opt_str = """#!/usr/bin/env python3

import os
import numpy as np
from ase.io import read
from ase.io.trajectory import TrajectoryWriter

'''
check if structure optimization worked well
if not, this script will generate a fake outcar
'''

def Get_Element_Num(elements):
    '''Using the Atoms.symples to Know Element&Num'''
    element = []
    ele = {}
    element.append(elements[0])
    for x in elements:
        if x not in element :
            element.append(x)
    for x in element:
        ele[x] = elements.count(x)
    return element, ele

def Write_Contcar(element, ele, lat, pos):
    '''Write CONTCAR'''
    f = open('CONTCAR','w')
    f.write('ASE-DP-FAILED\n')
    f.write('1.0\n')
    for i in range(3): 
        f.write('%15.10f %15.10f %15.10f\n' % tuple(lat[i])) 
    for x in element: 
        f.write(x + '  ')
    f.write('\n') 
    for x in element:
        f.write(str(ele[x]) + '  ')
    f.write('\n') 
    f.write('Direct\n')
    na = sum(ele.values())
    dpos = np.dot(pos,np.linalg.inv(lat))
    for i in range(na):
        f.write('%15.10f %15.10f %15.10f\n' % tuple(dpos[i]))

def Write_Outcar(element, ele, volume, lat, pos, ene, force, stress,pstress):
    '''Write OUTCAR'''
    f = open('OUTCAR','w')
    for x in element: 
        f.write('VRHFIN =' + str(x) + '\n')
    f.write('ions per type =')
    for x in element:
        f.write('%5d' % ele[x])
    f.write('\nDirection     XX             YY             ZZ             XY             YZ             ZX\n') 
    f.write('in kB')
    f.write('%15.6f' % stress[0])
    f.write('%15.6f' % stress[1])
    f.write('%15.6f' % stress[2])
    f.write('%15.6f' % stress[3])
    f.write('%15.6f' % stress[4])
    f.write('%15.6f' % stress[5])
    f.write('\n')
    ext_pressure = np.sum(stress[0] + stress[1] + stress[2])/3.0 - pstress
    f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\n'% (ext_pressure, pstress))
    f.write('volume of cell : %20.6f\n' % volume)
    f.write('direct lattice vectors\n')
    for i in range(3):
        f.write('%10.6f %10.6f %10.6f\n' % tuple(lat[i]))
    f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\n')
    f.write('-------------------------------------------------------------------\n')
    na = sum(ele.values())
    for i in range(na): 
        f.write('%15.6f %15.6f %15.6f' % tuple(pos[i]))
        f.write('%15.6f %15.6f %15.6f\n' % tuple(force[i]))
    f.write('-------------------------------------------------------------------\n')
    f.write('energy  without entropy= %20.6f %20.6f\n' % (ene, ene))
    enthalpy = ene + pstress * volume / 1602.17733 
    f.write('enthalpy is  TOTEN    = %20.6f %20.6f\n' % (enthalpy, enthalpy))

def check():
    to_be_opti = read('POSCAR')
    traj = TrajectoryWriter('traj.traj', 'w', to_be_opti)
    traj.write()
    traj.close()
    atoms_symbols_f = to_be_opti.get_chemical_symbols()
    element_f, ele_f = Get_Element_Num(atoms_symbols_f)
    atoms_vol_f = to_be_opti.get_volume()
    atoms_stress_f = np.array([0, 0, 0, 0, 0, 0])
    atoms_lat_f = to_be_opti.cell 
    atoms_pos_f = to_be_opti.positions
    atoms_force_f = np.zeros((atoms_pos_f.shape[0], 3))
    atoms_ene_f =  610612509
    Write_Contcar(element_f, ele_f, atoms_lat_f, atoms_pos_f)
    Write_Outcar(element_f, ele_f, atoms_vol_f, atoms_lat_f, atoms_pos_f,atoms_ene_f, atoms_force_f, atoms_stress_f * -10.0, 0)

if __name__ == "__main__":
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd,'OUTCAR')):
        check()"""