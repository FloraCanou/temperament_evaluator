# © 2020-2021 Flora Canou | Version 0.8
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from scipy import optimize, linalg
np.set_printoptions (suppress = True, linewidth = 256)

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SCALAR = 1200 #could be in octave, but for precision reason

def weighted (matrix, subgroup, type = "tenney"):
    if not type in {"tenney", "frobenius"}:
        type = "tenney"

    if type == "tenney":
        weighter = np.diag (1/np.log2 (subgroup))
    elif type == "frobenius":
        weighter = np.eye (len (subgroup))
    return matrix @ weighter

def error (gen, map, jip, order = 2):
    return linalg.norm (gen @ map - jip, ord = order)

def optimizer_main (map, subgroup = [], order = 2, weighter = "tenney", cons_monzo_list = np.array ([]), stretch_monzo = np.array ([]), show = True):
    if len (subgroup) == 0:
        subgroup = PRIME_LIST[:map.shape[1]]

    jip = np.log2 (subgroup)*SCALAR
    map_w = weighted (map, subgroup, type = weighter)
    jip_w = weighted (jip, subgroup, type = weighter)
    if order == 2 and not cons_monzo_list.size: #te with no constraints, simply use lstsq for better performance
        res = linalg.lstsq (map_w.T, jip_w)
        gen = res[0]
        print ("L2 tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [SCALAR]*map.shape[0] #initial guess
        cons = {'type': 'eq', 'fun': lambda gen: (gen @ map - jip) @ cons_monzo_list} if cons_monzo_list.size else ()
        res = optimize.minimize (error, gen0, args = (map_w, jip_w, order), method = "SLSQP", constraints = cons)
        print (res.message)
        if res.success:
            gen = res.x

    if stretch_monzo.size:
        gen *= (jip @ stretch_monzo)/(gen @ map @ stretch_monzo)

    if show:
        print (f"Generators: {gen} (¢)", f"Tuning map: {gen @ map} (¢)", sep = "\n")

    return gen
