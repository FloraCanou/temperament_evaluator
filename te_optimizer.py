# © 2020-2023 Flora Canou | Version 0.26.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def __error (gen, vals, jip, order):
    return linalg.norm (gen @ vals - jip, ord = order)

def optimizer_main (vals, subgroup = None, norm = te.Norm (), #"map" is a reserved word
        cons_monzo_list = None, des_monzo = None, show = True): 
    vals, subgroup = te.get_subgroup (vals, subgroup, axis = te.ROW)

    jip = np.log2 (subgroup)*te.SCALAR
    vals_wx = norm.weightskewed (vals, subgroup)
    jip_wx = norm.weightskewed (jip, subgroup)
    if norm.order == 2 and cons_monzo_list is None: #simply using lstsq for better performance
        res = linalg.lstsq (vals_wx.T, jip_wx)
        gen = res[0]
        print ("Euclidean tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [te.SCALAR]*vals.shape[0] #initial guess
        cons = () if cons_monzo_list is None else {'type': 'eq', 'fun': lambda gen: (gen @ vals - jip) @ cons_monzo_list}
        res = optimize.minimize (__error, gen0, args = (vals_wx, jip_wx, norm.order), method = "SLSQP",
            options = {'ftol': 1e-9}, constraints = cons)
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
            gen *= (jip @ des_monzo)/tempered_size

    tuning_map = gen @ vals
    mistuning_map = tuning_map - jip

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")

    return gen, tuning_map, mistuning_map

optimiser_main = optimizer_main
