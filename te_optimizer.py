# © 2020-2023 Flora Canou | Version 0.27.2
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def optimizer_main (breeds, subgroup = None, norm = te.Norm (), 
        cons_monzo_list = None, des_monzo = None, show = True): 
    """
    Returns the generator tuning map, tuning map, and error map. 
    The result can be displayed. 
    """
    # NOTE: "map" is a reserved word
    # optimization is preferably done in the unit of octaves, but for precision reasons

    breeds, subgroup = te.setup (breeds, subgroup, axis = te.AXIS.ROW)

    just_tuning_map = te.SCALAR.CENT*np.log2 (subgroup)
    breeds_x = norm.tuning_x (breeds, subgroup)
    just_tuning_map_x = norm.tuning_x (just_tuning_map, subgroup)
    if norm.order == 2 and cons_monzo_list is None: #simply using lstsq for better performance
        res = linalg.lstsq (breeds_x.T, just_tuning_map_x)
        gen = res[0]
        print ("Euclidean tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [te.SCALAR.CENT]*breeds.shape[0] #initial guess
        cons = () if cons_monzo_list is None else {
            'type': 'eq', 
            'fun': lambda gen: (gen @ breeds - just_tuning_map) @ cons_monzo_list
        }
        res = optimize.minimize (lambda gen: linalg.norm (gen @ breeds_x - just_tuning_map_x, ord = norm.order), gen0, 
            method = "SLSQP", options = {'ftol': 1e-9}, constraints = cons)
        print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if not des_monzo is None:
        if np.asarray (des_monzo).ndim > 1 and np.asarray (des_monzo).shape[1] != 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (tempered_size := gen @ breeds @ des_monzo) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            gen *= (just_tuning_map @ des_monzo)/tempered_size

    tempered_tuning_map = gen @ breeds
    error_map = tempered_tuning_map - just_tuning_map

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")

    return gen, tempered_tuning_map, error_map

optimiser_main = optimizer_main
