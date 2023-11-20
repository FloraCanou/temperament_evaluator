# © 2020-2023 Flora Canou | Version 0.26.3
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def __error (gen, vals, just_tuning_map, order):
    return linalg.norm (gen @ vals - just_tuning_map, ord = order)

def optimizer_main (vals, subgroup = None, norm = te.Norm (), 
        cons_monzo_list = None, des_monzo = None, show = True): 
    # NOTE: "map" is a reserved word
    # optimization is preferably done in the unit of octaves, but for precision reasons

    vals, subgroup = te.get_subgroup (vals, subgroup, axis = te.AXIS.ROW)

    just_tuning_map = te.SCALAR.CENT*np.log2 (subgroup)
    vals_x = norm.weightskewed (vals, subgroup)
    just_tuning_map_x = norm.weightskewed (just_tuning_map, subgroup)
    if norm.order == 2 and cons_monzo_list is None: #simply using lstsq for better performance
        res = linalg.lstsq (vals_x.T, just_tuning_map_x)
        gen = res[0]
        print ("Euclidean tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [te.SCALAR.CENT]*vals.shape[0] #initial guess
        cons = () if cons_monzo_list is None else {
            'type': 'eq', 
            'fun': lambda gen: (gen @ vals - just_tuning_map) @ cons_monzo_list
        }
        res = optimize.minimize (__error, gen0, args = (vals_x, just_tuning_map_x, norm.order), 
            method = "SLSQP", options = {'ftol': 1e-9}, constraints = cons)
        print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if not des_monzo is None:
        if np.asarray (des_monzo).ndim > 1 and np.asarray (des_monzo).shape[1] != 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (tempered_size := gen @ vals @ des_monzo) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            gen *= (just_tuning_map @ des_monzo)/tempered_size

    tempered_tuning_map = gen @ vals
    mistuning_map = tempered_tuning_map - just_tuning_map

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")

    return gen, tempered_tuning_map, mistuning_map

optimiser_main = optimizer_main
