# © 2020-2022 Flora Canou | Version 0.18.1
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def error (gen, map, jip, order = 2):
    return linalg.norm (gen @ map - jip, ord = order)

def optimizer_main (map, subgroup = None, wtype = "tenney", order = 2,
        cons_monzo_list = None, des_monzo = None, show = True):
    map, subgroup = te.subgroup_normalize (np.array (map), subgroup, axis = "row")

    if wtype == "weil" and order != 2:
        warnings.warn ("The weil weighter as of now is only meant to be used for L2.")

    jip = np.log2 (subgroup)*te.SCALAR
    map_w = te.weighted (map, subgroup, wtype = wtype)
    jip_w = te.weighted (jip, subgroup, wtype = wtype)
    if order == 2 and cons_monzo_list is None: #te with no constraints, simply use lstsq for better performance
        res = linalg.lstsq (map_w.T, jip_w)
        gen = res[0]
        print ("L2 tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [te.SCALAR]*map.shape[0] #initial guess
        cons = () if cons_monzo_list is None else {'type': 'eq', 'fun': lambda gen: (gen @ map - jip) @ cons_monzo_list}
        res = optimize.minimize (error, gen0, args = (map_w, jip_w, order), method = "SLSQP", constraints = cons)
        print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if not des_monzo is None:
        if np.array (des_monzo).ndim > 1 and np.array (des_monzo).shape[1] != 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (tempered_size := gen @ map @ des_monzo) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            gen *= (jip @ des_monzo)/tempered_size

    tuning_map = gen @ map

    if show:
        print (f"Generators: {gen} (¢)", f"Tuning map: {tuning_map} (¢)", sep = "\n")

    return gen, tuning_map

optimiser_main = optimizer_main
