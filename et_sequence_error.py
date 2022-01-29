# © 2020-2021 Flora Canou | Version 0.10
# This work is licensed under the GNU General Public License version 3.

import te_temperament_measures as tm
import numpy as np

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

# Et construction function
def et_construct (n, subgroup, alt_val = 0):
    val = np.rint (n*np.log2 (subgroup)).astype (int, copy = False) + alt_val
    return tm.Temperament ([val], subgroup)

# Finds et sequence from comma list. Can be used to find optimal patent vals
# Comma list should be entered as column vectors
def et_sequence_error (monzo_list = None, subgroup = None, cond = "error", ntype = "breed", wtype = "tenney", pv = False, prog = True, threshold = 20, search_range = 1200):
    if monzo_list is None:
        if subgroup is None:
            raise ValueError ("please specify a monzo list or a subgroup. ")
        else:
            monzo_list = np.transpose ([[0]*len (subgroup)])
    else:
        if subgroup is None:
            subgroup = PRIME_LIST[:monzo_list.shape[0]]
        elif len (subgroup) != monzo_list.shape[0]:
            raise IndexError ("dimension does not match. ")

    gpv = [0]*len (subgroup) #initialize with the all-zeroes val
    while 0 in gpv : #skip vals with zeroes
        gpv = find_next_gpv (gpv, subgroup)
    while gpv[0] <= search_range:
        if (not pv or is_pv (gpv, subgroup = subgroup)) and np.gcd.reduce (gpv) == 1 and not np.any (([gpv] @ monzo_list)):
        # is patent val (if pv is set) and defactored and tempering out the commas
            et = tm.Temperament ([gpv], subgroup)
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

# Checks if a val is a GPV
def is_gpv (val, subgroup = None):
    if subgroup is None:
        subgroup = PRIME_LIST[:len (val)]
    elif len (val) != len (subgroup):
        raise IndexError ("dimension does not match. ")

    lower_bounds = (np.array (val) - 0.5) / np.log2 (subgroup)
    upper_bounds = (np.array (val) + 0.5) / np.log2 (subgroup)
    return True if max (lower_bounds) < min (upper_bounds) else False

# Checks if a val is a patent val
def is_pv (val, subgroup = None):
    if subgroup is None:
        subgroup = PRIME_LIST[:len (val)]
    elif len (val) != len (subgroup):
        raise IndexError ("dimension does not match. ")

    return True if all (val == np.round (val[0]*np.log2 (subgroup))) else False

# Enter a GPV, finds the next one
# Doesn't handle some nontrivial subgroups
def find_next_gpv (gpv_current, subgroup = None):
    if subgroup is None:
        subgroup = PRIME_LIST[:len (gpv_current)]
    if not is_gpv (gpv_current, subgroup): #verify input
        raise ValueError ("input is not a GPV. ")

    for i in range (1, len (subgroup) + 1):
        gpv_candidate = list.copy (gpv_current)
        gpv_candidate[-i] += 1
        if is_gpv (gpv_candidate, subgroup):
            return gpv_candidate
    else:
        raise NotImplementedError ("this nontrivial subgroup cannot be processed. ")
