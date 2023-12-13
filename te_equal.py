# © 2020-2023 Flora Canou | Version 1.2.0
# This work is licensed under the GNU General Public License version 3.

import re, warnings
import numpy as np
import te_common as te
import te_temperament_measures as te_tm

WARTS_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x"]

def et_construct (et_list, subgroup):
    """Temperament construction function from equal temperaments."""
    breeds = np.array ([warts2breed (n, subgroup) for n in te.as_list (et_list)])
    return te_tm.Temperament (breeds, subgroup)

def comma_construct (monzos, subgroup = None):
    """Temperament construction function from commas."""
    monzos, subgroup = te.setup (monzos, subgroup, axis = te.AXIS.COL)
    breeds = te.antinullspace (monzos)
    return te_tm.Temperament (breeds, subgroup)

def et_sequence (monzos = None, subgroup = None, ntype = "breed", norm = te.Norm (), inharmonic = False, 
        cond = "error", pv = False, prog = True, threshold = 20, search_range = 1200):
    """
    Finds the optimal sequence from the comma list. 
    Can be used to find optimal PVs and/or GPVs. 
    Comma list should be entered as column vectors. 
    """
    if not norm.order == 2:
        raise NotImplementedError ("non-Euclidean norms not supported as of now. ")
    elif monzos is None:
        if subgroup is None:
            raise ValueError ("please specify a monzo list or a subgroup. ")
        else:
            monzos = np.zeros ((len (subgroup), 1))
    else:
        monzos, subgroup = te.setup (monzos, subgroup, axis = te.AXIS.COL)
    do_inharmonic = (inharmonic or subgroup.is_prime ()
        or norm.wtype == "tenney" and subgroup.is_prime_power ())
    if not do_inharmonic and subgroup.index () == np.inf and cond == "badness":
        raise ValueError ("this measure is only defined on nondegenerate subgroups. ")

    print ("\nOptimal GPV sequence: ")
    gpv_infra = [0]*len (subgroup) #initialize with the all-zeroes breed
    search_flag = 1
    while (gpv_infra := __gpv_roll (gpv_infra, subgroup, 1))[0] == 0: #skip zero-equave breeds
        gpv = gpv_infra
    while (gpv := __gpv_roll (gpv, subgroup, 1))[0] <= search_range:
        # notification at multiples of 1200
        if gpv[0] % te.SCALAR.CENT == 0 and gpv[0] / te.SCALAR.CENT == search_flag: 
            print (f"Currently searching: {gpv[0]}")
            search_flag += 1
        # condition of further analysis
        if (pv and not is_pv (gpv, subgroup) # non-patent val if pv is set
            or np.gcd.reduce (gpv) > 1 #enfactored
            or np.any ([gpv] @ monzos)): #not tempering out the commas
                continue

        et = te_tm.Temperament ([gpv], subgroup, saturate = False, normalize = False)
        if cond == "error":
            current = et._Temperament__error (ntype, norm, do_inharmonic, te.SCALAR.CENT)
        elif cond == "badness":
            current = et._Temperament__badness (ntype, norm, do_inharmonic, te.SCALAR.OCTAVE)
        else:
            current = threshold
        if current <= threshold:
            if prog:
                threshold = current
            print (f"{te.bra (gpv)} ({breed2warts (gpv, subgroup)})")
    print ("Search complete. ")

def is_gpv (breed, subgroup = None):
    """Checks if a breed is a GPV."""
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    just_tuning_map = subgroup.just_tuning_map ()
    lower_bounds = (np.asarray (breed) - 0.5) / just_tuning_map
    upper_bounds = (np.asarray (breed) + 0.5) / just_tuning_map
    return max (lower_bounds) < min (upper_bounds)

def is_pv (breed, subgroup = None):
    """Checks if a breed is a patent val."""
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    just_tuning_map = subgroup.just_tuning_map ()
    return all (breed == np.rint (breed[0]*just_tuning_map/just_tuning_map[0]))

def gpv_roll (breed, subgroup = None, n = 1):
    """
    Enter a GPV, finds the n-th next GPV. 
    Doesn't handle some nontrivial subgroups. 
    """
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    if not is_gpv (breed, subgroup): #verify input
        raise ValueError ("input is not a GPV. ")
    if not isinstance (n, (int, np.integer)):
        raise TypeError ("n must be an integer. ")
    return __gpv_roll (breed, subgroup, n)

def __gpv_roll (breed, subgroup, n):
    if n == 0:
        return breed
    else:
        for i in range (1, len (subgroup) + 1):
            breed_copy = np.array (breed)
            breed_copy[-i] += np.copysign (1, n).astype (int)
            if is_gpv (breed_copy, subgroup):
                return __gpv_roll (breed_copy, subgroup, n - np.copysign (1, n).astype (int))
        else:
            raise NotImplementedError ("this nontrivial subgroup cannot be processed. ")

def __just_tuning_map_n (n, equave, subgroup):
    """Finds the just tuning map in terms of edostep numbers of n-ed-p."""
    just_tuning_map = subgroup.just_tuning_map ()
    return n*just_tuning_map/np.log2 (equave)

def breed2warts (breed, subgroup = None):
    """
    Enter a breed, finds its wart notation. 
    Zero equave is disallowed. 
    """
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    if breed[0] == 0:
        raise ValueError ("Wart is undefined. ")

    equave = subgroup.ratios ()[0].value ()
    if equave == 2: #octave equave
        prefix = ""
    elif equave in te.PRIME_LIST: #non-octave prime equave
        prefix = str (WARTS_LIST[te.PRIME_LIST.index (equave.value ())])
    else: #nonprime equave
        prefix = "*"

    if is_pv (breed, subgroup): #patent val
        postfix = ""
    elif all (entry in te.PRIME_LIST for entry in subgroup.ratios (evaluate = True)): #nonpatent val in prime subgroup
        just_tuning_map_n = __just_tuning_map_n (breed[0], equave, subgroup) #just tuning map in edostep numbers
        pv = np.rint (just_tuning_map_n) #corresponding patent val
        warts_number_list = (
            2*np.abs (breed - pv) + (np.copysign (1, (breed - pv)*(pv - just_tuning_map_n)) - 1)/2
            ).astype (int)
        postfix = ""
        for i, si in enumerate (subgroup.ratios (evaluate = True)):
            postfix += warts_number_list[i]*str (WARTS_LIST[te.PRIME_LIST.index (si)])
    else: #nonpatent val in nonprime subgroup
        postfix = "*"

    return prefix + str (breed[0]) + postfix

def warts2breed (warts, subgroup):
    """
    Enter a wart notation and a subgroup, finds the breed. 
    Same letter in the prefix and postfix is considered invalid. 
    Equave is the octave regardless of the subgroup
    unless specified explicitly with warts. 
    """
    if isinstance (warts, str):
        return __warts2breed (warts, subgroup)
    elif isinstance (warts, (int, float)):
        return np.rint (__just_tuning_map_n (warts, 2, subgroup)).astype (int)
    else:
        raise TypeError ("Enter a string or number. ")

def __warts2breed (warts, subgroup):
    match = re.match ("(^[a-x]?)(\d+)([a-x]*)", str (warts))
    if not match or (match.group (1) and re.match (match.group (1), match.group (3))):
        raise ValueError ("Invalid wart notation. ")

    wart_equave = te.PRIME_LIST[WARTS_LIST.index (match.group (1))] if match.group (1) else 2
    warts_number_list = np.zeros (len (subgroup))
    for i, si in enumerate (subgroup.ratios (evaluate = True)):
        if si in te.PRIME_LIST:
            warts_number_list[i] = len (re.findall (WARTS_LIST[te.PRIME_LIST.index (si)], match.group (3)))
    just_tuning_map_n = __just_tuning_map_n (int (match.group (2)), wart_equave, subgroup) #just tuning map in edostep numbers
    pv = np.rint (just_tuning_map_n) #corresponding patent val
    alt_breed = np.copysign (np.ceil (warts_number_list/2), (1 - 2*(warts_number_list % 2))*(pv - just_tuning_map_n))

    return (pv + alt_breed).astype (int)
