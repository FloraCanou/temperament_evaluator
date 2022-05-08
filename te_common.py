# © 2020-2022 Flora Canou | Version 0.18.2
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import linalg

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SCALAR = 1200 #could be in octave, but for precision reason

def subgroup_normalize (main, subgroup, axis):
    if axis == "row":
        length_main = main.shape[1]
    elif axis == "col":
        length_main = main.shape[0]
    elif axis == "vec":
        length_main = len (main)

    if subgroup is None:
        subgroup = PRIME_LIST[:length_main]
    elif length_main != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (length_main, len (subgroup))
        if axis == "row":
            main = main[:, :dim]
        elif axis == "col":
            main = main[:dim, :]
        elif axis == "vec":
            main = main[:dim]
        subgroup = subgroup[:dim]

    return main, subgroup

def weighted (main, subgroup, wtype = "tenney"):
    if not wtype in {"tenney", "frobenius", "inverse tenney", "benedetti", "weil"}:
        warnings.warn ("unknown weighter type, using default (\"tenney\")")
        wtype = "tenney"

    if wtype == "tenney":
        weighter = np.diag (1/np.log2 (subgroup))
    elif wtype == "frobenius":
        weighter = np.eye (len (subgroup))
    elif wtype == "inverse tenney":
        weighter = np.diag (np.log2 (subgroup))
    elif wtype == "benedetti":
        weighter = np.diag (1/np.array (subgroup))
    elif wtype == "weil":
        weighter = linalg.pinv (np.append (np.diag (np.log2 (subgroup)), [np.log2 (subgroup)], axis = 0)/2)

    return main @ weighter

# takes a monzo, returns the ratio in [num, den] form
# doesn't validate the basis
# ratio[0]: num, ratio[1]: den
def monzo2ratio (monzo, subgroup = None):
    monzo, subgroup = subgroup_normalize (monzo, subgroup, axis = "vec")
    ratio = [1, 1]
    for i, mi in enumerate (monzo):
        if mi > 0:
            ratio[0] *= subgroup[i]**mi
        elif mi < 0:
            ratio[1] *= subgroup[i]**(-mi)
    return ratio
