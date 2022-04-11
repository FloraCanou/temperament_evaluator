# © 2020-2022 Flora Canou | Version 0.12.2
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from scipy import optimize, linalg
import warnings
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SCALAR = 1200 #could be in octave, but for precision reason

def subgroup_normalize (main, subgroup):
    if subgroup is None:
        subgroup = PRIME_LIST[:main.shape[1]]
    elif main.shape[1] != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (main.shape[1], len (subgroup))
        main = main[:, :dim]
        subgroup = subgroup[:dim]
    return main, subgroup

def weighted (matrix, subgroup, wtype = "tenney"):
    if not wtype in {"tenney", "frobenius", "partch"}:
        wtype = "tenney"
        warnings.warn ("unknown weighter type, using default (\"tenney\")")

    if wtype == "tenney":
        weighter = np.diag (1/np.log2 (subgroup))
    elif wtype == "frobenius":
        weighter = np.eye (len (subgroup))
    elif wtype == "partch":
        weighter = np.diag (np.log2 (subgroup))
    return matrix @ weighter

def error (gen, map, jip, order = 2):
    return linalg.norm (gen @ map - jip, ord = order)

def optimizer_main (map, subgroup = None, wtype = "tenney", order = 2, cons_monzo_list = None, stretch_monzo = None, show = True):
    map, subgroup = subgroup_normalize (np.array (map), subgroup)

    jip = np.log2 (subgroup)*SCALAR
    map_w = weighted (map, subgroup, wtype = wtype)
    jip_w = weighted (jip, subgroup, wtype = wtype)
    if order == 2 and cons_monzo_list is None: #te with no constraints, simply use lstsq for better performance
        res = linalg.lstsq (map_w.T, jip_w)
        gen = res[0]
        print ("L2 tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [SCALAR]*map.shape[0] #initial guess
        cons = () if cons_monzo_list is None else {'type': 'eq', 'fun': lambda gen: (gen @ map - jip) @ cons_monzo_list}
        res = optimize.minimize (error, gen0, args = (map_w, jip_w, order), method = "SLSQP", constraints = cons)
        print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if not stretch_monzo is None:
        if stretch_monzo.shape[1] != 1:
            raise IndexError ("only one stretch target is allowed. ")
        elif (tempered_size := gen @ map @ stretch_monzo) == 0:
            raise ZeroDivisionError ("stretch target is in the nullspace. ")
        else:
            gen *= (jip @ stretch_monzo)/tempered_size

    tuning_map = gen @ map

    if show:
        print (f"Generators: {gen} (¢)", f"Tuning map: {tuning_map} (¢)", sep = "\n")

    return gen, tuning_map

optimiser_main = optimizer_main