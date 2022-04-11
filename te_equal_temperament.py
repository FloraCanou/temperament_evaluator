# © 2020-2022 Flora Canou | Version 0.13
# This work is licensed under the GNU General Public License version 3.

import numpy as np
import warnings
import te_common as te
import te_temperament_measures as te_tm

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

# Temperament construction function from ets
def et_construct (et_list, subgroup, alt_val = 0):
    try:
        map = np.array ([n*np.log2 (subgroup) for n in et_list]) + np.resize (alt_val, (len (et_list), len (subgroup)))
    except TypeError:
        map = np.array ([et_list*np.log2 (subgroup)]) + np.resize (alt_val, (1, len (subgroup)))
        warnings.warn ("equal temperament number should be entered as a list. ", FutureWarning)
    return te_tm.Temperament (map, subgroup)

# Finds et sequence from comma list. Can be used to find optimal patent vals
# Comma list should be entered as column vectors
def et_sequence (monzo_list = None, subgroup = None, cond = "error", ntype = "breed", wtype = "tenney", pv = False, prog = True, threshold = 20, search_range = 1200):
    if monzo_list is None:
        if subgroup is None:
            raise ValueError ("please specify a monzo list or a subgroup. ")
        else:
            monzo_list = np.zeros ((len (subgroup), 1))
    else:
        monzo_list, subgroup = te.subgroup_normalize (monzo_list, subgroup, axis = "col")

    gpv = [0]*len (subgroup) #initialize with the all-zeroes val
    while 0 in gpv : #skip vals with zeroes
        gpv = find_next_gpv (gpv, subgroup)
    while gpv[0] <= search_range:
        if (not pv or is_pv (gpv, subgroup = subgroup)) and np.gcd.reduce (gpv) == 1 and not np.any (([gpv] @ monzo_list)):
        # is patent val (if pv is set) and defactored and tempering out the commas
            et = te_tm.Temperament ([gpv], subgroup)
            if cond == "error":
                if (current := et.error (ntype = ntype, wtype = wtype)) <= threshold:
                    if prog:
                        threshold = current
                    et.temperament_measures (ntype = ntype, wtype = wtype)
            elif cond == "badness":
                if (current := et.badness (ntype = ntype, wtype = wtype)) <= threshold:
                    if prog:
                        threshold = current
                    et.temperament_measures (ntype = ntype, wtype = wtype)
            else:
                et.temperament_measures (ntype = ntype, wtype = wtype)
        gpv = find_next_gpv (gpv, subgroup)

et_sequence_error = et_sequence

# Checks if a val is a GPV
def is_gpv (val, subgroup = None):
    val, subgroup = te.subgroup_normalize (val, subgroup, axis = "vec")
    lower_bounds = (np.array (val) - 0.5) / np.log2 (subgroup)
    upper_bounds = (np.array (val) + 0.5) / np.log2 (subgroup)
    return True if max (lower_bounds) < min (upper_bounds) else False

# Checks if a val is a patent val
def is_pv (val, subgroup = None):
    val, subgroup = te.subgroup_normalize (val, subgroup, axis = "vec")
    return True if all (val == np.round (val[0]*np.log2 (subgroup))) else False

# Enter a GPV, finds the next one
# Doesn't handle some nontrivial subgroups
def find_next_gpv (val, subgroup = None):
    val, subgroup = te.subgroup_normalize (val, subgroup, axis = "vec")
    if not is_gpv (val, subgroup): #verify input
        raise ValueError ("input is not a GPV. ")

    for i in range (1, len (subgroup) + 1):
        val_copy = list.copy (val)
        val_copy[-i] += 1
        if is_gpv (val_copy, subgroup):
            return val_copy
    else:
        raise NotImplementedError ("this nontrivial subgroup cannot be processed. ")

def warts2val (warts, subgroup = None):
    pass

def val2warts (val, subgroup = None):
    pass
