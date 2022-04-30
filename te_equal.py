# © 2020-2022 Flora Canou | Version 0.16.1
# This work is licensed under the GNU General Public License version 3.

import numpy as np
import re, warnings
import te_common as te
import te_temperament_measures as te_tm

WARTS_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]

# Temperament construction function from ets
def et_construct (et_list, subgroup, alt_val = 0):
    if isinstance (et_list, list):
        map = np.array ([warts2val (n, subgroup) for n in et_list]) + np.resize (alt_val, (len (et_list), len (subgroup)))
    else:
        map = np.array ([warts2val (et_list, subgroup)]) + np.resize (alt_val, (1, len (subgroup)))
        warnings.warn ("equal temperament number should be entered as a list. ", FutureWarning)
    return te_tm.Temperament (map, subgroup)

# Finds et sequence from comma list. Can be used to find optimal patent vals
# Comma list should be entered as column vectors
def et_sequence (monzo_list = None, subgroup = None, cond = "error", ntype = "breed", wtype = "tenney", pv = False, prog = True, verbose = False, threshold = 20, search_range = 1200):
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
                    if verbose:
                        et.temperament_measures (ntype = ntype, wtype = wtype)
                    else:
                        print (f"{gpv} ({val2warts (gpv, subgroup = subgroup)})")
            elif cond == "badness":
                if (current := et.badness (ntype = ntype, wtype = wtype)) <= threshold:
                    if prog:
                        threshold = current
                    if verbose:
                        et.temperament_measures (ntype = ntype, wtype = wtype)
                    else:
                        print (f"{gpv} ({val2warts (gpv, subgroup = subgroup)})")
            else:
                if verbose:
                    et.temperament_measures (ntype = ntype, wtype = wtype)
                else:
                    print (f"{gpv} ({val2warts (gpv, subgroup = subgroup)})")
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
    return True if all (val == np.round (val[0]*np.log2 (subgroup)/np.log2 (subgroup[0]))) else False

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

# Enter a val, finds its wart notation
def val2warts (val, subgroup = None):
    val, subgroup = te.subgroup_normalize (val, subgroup, axis = "vec")

    if subgroup[0] == 2: #octave equave
        prefix = ""
    elif subgroup[0] in te.PRIME_LIST: #non-octave prime equave
        prefix = str (WARTS_LIST[te.PRIME_LIST.index (subgroup[0])])
    else: #nonprime equave
        prefix = "*"
    if is_pv (val, subgroup):
        postfix = ""
    else:
        if all (entry in te.PRIME_LIST for entry in subgroup): #prime subgroup
            jip_n = val[0]*np.log2 (subgroup)/np.log2 (subgroup[0]) #jip in edostep numbers
            pv = np.round (jip_n) #corresponding patent val
            warts_number_list = (2*np.abs (val - pv) + (np.copysign (1, (val - pv)*(pv - jip_n)) - 1)/2).astype ("int")
            postfix = ""
            for i in range (len (val)):
                postfix += warts_number_list[i]*str (WARTS_LIST[te.PRIME_LIST.index (subgroup[i])])
        else: #nonprime subgroup
            postfix = "*"
    return prefix + str (val[0]) + postfix

# Enter a wart notation and a subgroup, finds the val
# Presence of the same letter in the prefix and postfix is considered invalid
# Equave is the octave regardless of the subgroup unless specified explicitly
def warts2val (warts, subgroup):
    match = re.match ("(^[a-r]?)(\d+)([a-r]*)", str (warts))
    if not match or (match.group (1) and re.match (match.group (1), match.group (3))):
        raise ValueError ("Invalid wart notation. ")

    wart_equave = te.PRIME_LIST[WARTS_LIST.index (match.group (1))] if match.group (1) else 2
    warts_number_list = np.array ([len (re.findall (WARTS_LIST[te.PRIME_LIST.index (entry)], match.group (3))) for entry in subgroup])
    jip_n = int (match.group (2))*np.log (subgroup)/np.log (wart_equave) #jip in edostep numbers
    pv = np.round (jip_n) #corresponding patent val
    alt_val = np.copysign (np.ceil (warts_number_list/2), (1 - 2*(warts_number_list % 2))*(pv - jip_n))
    return (pv + alt_val).astype ("int")
