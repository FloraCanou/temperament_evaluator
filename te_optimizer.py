# © 2020-2022 Flora Canou | Version 0.21.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def error (gen, map, jip, order):
    return linalg.norm (gen @ map - jip, ord = order)

def optimizer_main (map, subgroup = None, wtype = "tenney", skew = 0, order = 2,
        cons_monzo_list = None, des_monzo = None, show = True):
    map, subgroup = te.get_subgroup (np.array (map), subgroup, axis = te.ROW)

    jip = np.log2 (subgroup)*te.SCALAR
    map_wx = te.weightskewed (map, subgroup, wtype, skew, order)
    jip_wx = te.weightskewed (jip, subgroup, wtype, skew, order)
    if order == 2 and cons_monzo_list is None: #te with no constraints, simply use lstsq for better performance
        res = linalg.lstsq (map_wx.T, jip_wx)
        gen = res[0]
        print ("L2 tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [te.SCALAR]*map.shape[0] #initial guess
        cons = () if cons_monzo_list is None else {'type': 'eq', 'fun': lambda gen: (gen @ map - jip) @ cons_monzo_list}
        res = optimize.minimize (error, gen0, args = (map_wx, jip_wx, order), method = "SLSQP",
            options = {'ftol': 1e-9}, constraints = cons)
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
    mistuning_map = tuning_map - jip

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")

    return gen, tuning_map, mistuning_map

optimiser_main = optimizer_main
