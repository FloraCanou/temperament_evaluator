# © 2020-2021 Flora Canou | Version 0.6
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
def et_sequence_error (monzo_list, subgroup = [], cond = "error", progressive = True, threshold = 20, search_range = 1200):
    if len (subgroup) == 0:
        subgroup = PRIME_LIST[:monzo_list.shape[0]]
    gpv = [0]*len (subgroup) #initialize with the all-zeroes val
    while not all (gpv): #skip vals with zeroes
        gpv = find_next_gpv (gpv, subgroup)
    while gpv[0] <= search_range:
        if not any (([gpv] @ monzo_list)[0]): #tempering out the commas
            et = tm.Temperament ([gpv], subgroup)
            if cond == "error":
                if et.error () <= threshold:
                    if progressive:
                        threshold = et.error ()
                    et.temperament_measures ()
            elif cond == "badness":
                if et.badness () <= threshold:
                    if progressive:
                        threshold = et.badness ()
                    et.temperament_measures ()
            else:
                et.temperament_measures ()
        gpv = find_next_gpv (gpv, subgroup)

# Checks if a val is a GPV
def is_gpv (val, subgroup = []):
    if len (subgroup) == 0:
        subgroup = PRIME_LIST[:len (val)]
    elif len (val) != len (subgroup):
        raise IndexError ("dimension does not match. ")
    lower_bounds = (np.array (val) - 0.5) / np.log2 (subgroup)
    upper_bounds = (np.array (val) + 0.5) / np.log2 (subgroup)
    return True if max (lower_bounds) < min (upper_bounds) else False

# Enter a GPV, finds the next one
# Doesn't handle some nontrivial subgroups
def find_next_gpv (gpv_current, subgroup = []):
    if len (subgroup) == 0:
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
