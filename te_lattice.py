# © 2020-2022 Flora Canou | Version 0.19
# This work is licensed under the GNU General Public License version 3.

import math
import numpy as np
from scipy import linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256)

def find_temperamental_norm (map, monzo, subgroup = None, wtype = "tenney", oe = False, show = True):
    map, subgroup = te.subgroup_normalize (map, subgroup, axis = "row")
    monzo, subgroup = te.subgroup_normalize (monzo, subgroup, axis = "vec")

    if oe: #octave equivalence
        map = map[1:]
    projection_w = linalg.pinv (te.weighted (map, subgroup, wtype = wtype) @ te.weighted (map, subgroup, wtype = wtype).T)
    tmonzo = map @ monzo
    norm = np.sqrt (tmonzo.T @ projection_w @ tmonzo)
    if show:
        ratio = te.monzo2ratio (monzo, subgroup)
        print (f"{ratio[0]}/{ratio[1]}\t {norm}")
    return norm

def find_spectrum (map, monzo_list, subgroup = None, wtype = "tenney", oe = True):
    map, subgroup = te.subgroup_normalize (map, subgroup, axis = "row")
    monzo_list, subgroup = te.subgroup_normalize (monzo_list, subgroup, axis = "col")

    spectrum = [[monzo_list[:, i], find_temperamental_norm (map, monzo_list[:, i], subgroup = subgroup, wtype = wtype, oe = oe, show = False)] for i in range (monzo_list.shape[1])]
    spectrum.sort (key = lambda k: k[1])
    print ("\nComplexity spectrum: ")
    for entry in spectrum:
        ratio = te.monzo2ratio (entry[0], subgroup)
        print (f"{ratio[0]}/{ratio[1]}\t{entry[1]:.4f}")

# octave-reduce the ratio in [num, den] form
def ratio_8ve_reduction (ratio):
    oct = math.floor (math.log (ratio[0]/ratio[1], 2))
    if oct > 0:
        ratio[1] *= 2**oct
    elif oct < 0:
        ratio[0] *= 2**(-oct)
    return ratio

# enter an odd limit, returns the monzo list
def odd_limit_monzo_list_gen (odd_limit, sort = None):
    subgroup = list (filter (lambda q: q <= odd_limit, te.PRIME_LIST))
    ratio_list = []
    for num in range (1, odd_limit + 1, 2):
        for den in range (1, odd_limit + 1, 2):
            if math.gcd (num, den) == 1:
                ratio_list.append (ratio_8ve_reduction ([num, den]))
    if sort == "size":
        ratio_list.sort (key = lambda k: k[0]/k[1])
    return np.transpose ([te.ratio2monzo (entry, subgroup) for entry in ratio_list])
