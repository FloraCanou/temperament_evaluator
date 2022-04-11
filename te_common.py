# © 2020-2022 Flora Canou | Version 0.13
# This work is licensed under the GNU General Public License version 3.

import numpy as np
import warnings

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

def subgroup_normalize (main, subgroup, axis):
    if subgroup is None:
        if axis == "row":
            subgroup = PRIME_LIST[:main.shape[1]]
        elif axis == "col":
            subgroup = PRIME_LIST[:main.shape[0]]
        elif axis == "vec":
            subgroup = PRIME_LIST[:len (main)]
    elif axis == "row" and main.shape[1] != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (main.shape[1], len (subgroup))
        main = main[:, :dim]
        subgroup = subgroup[:dim]
    elif axis == "col" and main.shape[0] != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (main.shape[0], len (subgroup))
        main = main[:dim]
        subgroup = subgroup[:dim]
    elif axis == "vec" and len (main) != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (len (main), len (subgroup))
        main = main[:dim]
        subgroup = subgroup[:dim]
    return main, subgroup

def weighted (main, subgroup, wtype = "tenney"):
    if not wtype in {"tenney", "frobenius", "partch"}:
        wtype = "tenney"
        warnings.warn ("unknown weighter type, using default (\"tenney\")")

    if wtype == "tenney":
        weighter = np.diag (1/np.log2 (subgroup))
    elif wtype == "frobenius":
        weighter = np.eye (len (subgroup))
    elif wtype == "partch":
        weighter = np.diag (np.log2 (subgroup))
    return main @ weighter
