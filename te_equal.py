# © 2020-2023 Flora Canou | Version 0.24.1
# This work is licensed under the GNU General Public License version 3.

import re
import numpy as np
from sympy import Matrix
import te_common as te
import te_temperament_measures as te_tm

WARTS_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x"]

# Temperament construction function from ets
def et_construct (et_list, subgroup, alt_val = 0):
    val_list = (np.array ([warts2val (n, subgroup) for n in te.as_list (et_list)])
        + np.resize (alt_val, (len (te.as_list (et_list)), len (subgroup))))
    return te_tm.Temperament (val_list, subgroup)

# Temperament construction function from commas
def comma_construct (comma_list, subgroup = None):
    comma_list, subgroup = te.get_subgroup (comma_list, subgroup, axis = te.COL)
    val_list_frac = Matrix (np.flip (comma_list.T)).nullspace ()
    val_list = np.flip (np.row_stack ([te.matrix2array (entry) for entry in val_list_frac]))
    return te_tm.Temperament (val_list, subgroup)

# Finds et sequence from comma list. Can be used to find optimal patent vals
# Comma list should be entered as column vectors
def et_sequence (monzo_list = None, subgroup = None, cond = "error",
        ntype = "breed", wtype = "tenney", wamount = 1, skew = 0,
        pv = False, prog = True, threshold = 20, search_range = 1200):
    if monzo_list is None:
        if subgroup is None:
            raise ValueError ("please specify a monzo list or a subgroup. ")
        else:
            monzo_list = np.zeros ((len (subgroup), 1))
    else:
        monzo_list, subgroup = te.get_subgroup (monzo_list, subgroup, axis = te.COL)

    print ("\nOptimal GPV sequence: ")
    gpv_infra = [0]*len (subgroup) #initialize with the all-zeroes val
    search_flag = 1
    while (gpv_infra := __gpv_roll (gpv_infra, subgroup, 1))[0] == 0: #skip zero-equave vals
        gpv = gpv_infra
    while (gpv := __gpv_roll (gpv, subgroup, 1))[0] <= search_range:
        # notification at multiples of 1200
        if gpv[0] % te.SCALAR == 0 and gpv[0] / te.SCALAR == search_flag: 
            print (f"Currently searching: {gpv[0]}")
            search_flag += 1
        # condition of further analysis
        if (pv and not is_pv (gpv, subgroup) # non-patent val if pv is set
            or np.gcd.reduce (gpv) > 1 #enfactored
            or np.any ([gpv] @ monzo_list)): #not tempering out the commas
                continue

        et = te_tm.Temperament ([gpv], subgroup, saturate = False, normalize = False)
        if cond == "error":
            current = et.error (ntype, wtype, wamount, skew)
        elif cond == "badness":
            current = et.badness (ntype, wtype, wamount, skew)
        else:
            current = threshold
        if current <= threshold:
            if prog:
                threshold = current
            print (f"{te.bra (gpv)} ({val2warts (gpv, subgroup)})")
    print ("Search complete. ")

# Checks if a val is a GPV
def is_gpv (val, subgroup = None):
    val, subgroup = te.get_subgroup (val, subgroup, axis = te.VEC)
    lower_bounds = (np.asarray (val) - 0.5) / np.log2 (subgroup)
    upper_bounds = (np.asarray (val) + 0.5) / np.log2 (subgroup)
    return max (lower_bounds) < min (upper_bounds)

# Checks if a val is a patent val
def is_pv (val, subgroup = None):
    val, subgroup = te.get_subgroup (val, subgroup, axis = te.VEC)
    return all (val == np.rint (val[0]*np.log2 (subgroup)/np.log2 (subgroup[0])))

# Enter a GPV, finds the n-th next GPV
# Doesn't handle some nontrivial subgroups
def gpv_roll (val, subgroup = None, n = 1):
    val, subgroup = te.get_subgroup (val, subgroup, axis = te.VEC)
    if not is_gpv (val, subgroup): #verify input
        raise ValueError ("input is not a GPV. ")
    if not isinstance (n, (int, np.integer)):
        raise TypeError ("n must be an integer. ")
    return __gpv_roll (val, subgroup, n)

def __gpv_roll (val, subgroup, n):
    if n == 0:
        return val
    else:
        for i in range (1, len (subgroup) + 1):
            val_copy = list.copy (val)
            val_copy[-i] += np.copysign (1, n).astype (int)
            if is_gpv (val_copy, subgroup):
                return __gpv_roll (val_copy, subgroup, n - np.copysign (1, n).astype (int))
        else:
            raise NotImplementedError ("this nontrivial subgroup cannot be processed. ")

# Enter a val, finds its wart notation
# Zero equave is disallowed
def val2warts (val, subgroup = None):
    val, subgroup = te.get_subgroup (val, subgroup, axis = te.VEC)
    if val[0] == 0:
        raise ValueError ("Wart is undefined. ")

    if subgroup[0] == 2: #octave equave
        prefix = ""
    elif subgroup[0] in te.PRIME_LIST: #non-octave prime equave
        prefix = str (WARTS_LIST[te.PRIME_LIST.index (subgroup[0])])
    else: #nonprime equave
        prefix = "*"

    if is_pv (val, subgroup): #patent val
        postfix = ""
    elif all (entry in te.PRIME_LIST for entry in subgroup): #nonpatent val in prime subgroup
        jip_n = val[0]*np.log2 (subgroup)/np.log2 (subgroup[0]) #jip in edostep numbers
        pv = np.rint (jip_n) #corresponding patent val
        warts_number_list = (2*np.abs (val - pv) + (np.copysign (1, (val - pv)*(pv - jip_n)) - 1)/2).astype (int)
        postfix = ""
        for i, si in enumerate (subgroup):
            postfix += warts_number_list[i]*str (WARTS_LIST[te.PRIME_LIST.index (si)])
    else: #nonpatent val in nonprime subgroup
        postfix = "*"

    return prefix + str (val[0]) + postfix

# Enter a wart notation and a subgroup, finds the val
# Presence of the same letter in the prefix and postfix is considered invalid
# Equave is the octave regardless of the subgroup unless specified explicitly
def warts2val (warts, subgroup):
    match = re.match ("(^[a-x]?)(\d+)([a-x]*)", str (warts))
    if not match or (match.group (1) and re.match (match.group (1), match.group (3))):
        raise ValueError ("Invalid wart notation. ")

    wart_equave = te.PRIME_LIST[WARTS_LIST.index (match.group (1))] if match.group (1) else 2
    warts_number_list = np.zeros (len (subgroup))
    for i, si in enumerate (subgroup):
        if si in te.PRIME_LIST:
            warts_number_list[i] = len (re.findall (WARTS_LIST[te.PRIME_LIST.index (si)], match.group (3)))
    jip_n = int (match.group (2))*np.log (subgroup)/np.log (wart_equave) #jip in edostep numbers
    pv = np.rint (jip_n) #corresponding patent val
    alt_val = np.copysign (np.ceil (warts_number_list/2), (1 - 2*(warts_number_list % 2))*(pv - jip_n))

    return (pv + alt_val).astype (int)
