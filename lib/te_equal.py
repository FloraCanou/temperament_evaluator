# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import re, warnings
import numpy as np
from tqdm import tqdm
from . import te_common as te
from . import te_temperament_measures as te_tm

WARTS_DICT = {
    2: "a", 3: "b", 5: "c", 7: "d", 11: "e", 13: "f", 
    17: "g", 19: "h", 23: "i", 29: "j", 31: "k", 37: "l", 
    41: "m", 43: "n", 47: "o", 53: "p", 59: "q", 61: "r", 
    67: "s", 71: "t", 73: "u", 79: "v", 83: "w", 89: "x"
}

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
    if monzos is None:
        if subgroup is None:
            raise ValueError ("please specify a monzo list or a subgroup. ")
        else:
            monzos = np.zeros ((len (subgroup), 1))
    else:
        monzos, subgroup = te.setup (monzos, subgroup, axis = te.AXIS.COL)
    do_inharmonic = (inharmonic or subgroup.is_prime ()
        or norm.wmode == 1 and norm.wstrength == 1 and subgroup.is_prime_power ())
    if not do_inharmonic and subgroup.index () == np.inf and cond == "badness":
        raise ValueError ("this measure is only defined on nondegenerate subgroups. ")

    def __gpv_roll (breed, tuning_map):
        for i in range (1, len (tuning_map) + 1):
            breed_copy = breed.copy ()
            breed_copy[-i] += 1
            if __is_gpv (breed_copy, tuning_map):
                return breed_copy
        else:
            raise NotImplementedError ("this nontrivial tuning map cannot be processed. ")

    print ("\nOptimal GPV sequence: ")
    just_tuning_map = subgroup.just_tuning_map ()
    gpv = np.zeros (len (just_tuning_map), dtype = int) #initialize with the all-zeroes breed
    while (gpv := __gpv_roll (gpv, just_tuning_map))[0] == 0: #skip zero-equave breeds
        pass
    with tqdm (total = search_range) as progress_bar:
        search_flag = 1
        while gpv[0] <= search_range:
            if ((not pv or __is_pv (gpv, just_tuning_map)) # patent val or pv isn't set
                    and np.gcd.reduce (gpv) == 1 #not enfactored
                    and not np.any (gpv[np.newaxis] @ monzos)): #tempering out the commas
                match cond:
                    case "error":
                        et = te_tm.Temperament (gpv[np.newaxis], subgroup, saturate = False, normalize = False)
                        current = et._Temperament__error (ntype, norm, do_inharmonic, te.SCALAR.CENT)
                    case "badness":
                        et = te_tm.Temperament (gpv[np.newaxis], subgroup, saturate = False, normalize = False)
                        current = et._Temperament__badness (ntype, norm, do_inharmonic, te.SCALAR.OCTAVE)
                    case "logflat badness":
                        et = te_tm.Temperament (gpv[np.newaxis], subgroup, saturate = False, normalize = False)
                        current = et._Temperament__badness_logflat (ntype, norm, do_inharmonic, te.SCALAR.OCTAVE)
                    case _:
                        current = threshold
                if current <= threshold:
                    progress_bar.write (f"{te.bra (gpv)} ({breed2warts (gpv, subgroup)})")
                    if prog:
                        threshold = current
            
            # update the progress bar
            if gpv[0] == search_flag: 
                progress_bar.update ()
                search_flag += 1
            
            # roll to the next gpv
            gpv = __gpv_roll (gpv, just_tuning_map)
    print ("Search complete. ")

def is_gpv (breed, subgroup = None):
    """Checks if a breed is a GPV on a subgroup."""
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    just_tuning_map = subgroup.just_tuning_map ()
    return __is_gpv (breed, just_tuning_map)

def __is_gpv (breed, tuning_map):
    """
    Checks if a breed is a GPV for an arbitrary tuning map. 
    To be used in gpv_roll to avoid repeatedly computing the just tuning map, 
    which is expensive.
    """
    lower_bounds = (breed - 0.5) / tuning_map
    upper_bounds = (breed + 0.5) / tuning_map
    return max (lower_bounds) < min (upper_bounds)

def is_pv (breed, subgroup = None):
    """Checks if a breed is a patent val on a subgroup."""
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    just_tuning_map = subgroup.just_tuning_map ()
    return __is_pv (breed, just_tuning_map)

def __is_pv (breed, tuning_map):
    """
    Checks if a breed is a patent val for an arbitrary tuning map. 
    To be used in et_sequence to avoid repeatedly computing the just tuning map, 
    which is expensive.
    """
    return all (breed == np.rint (breed[0]*tuning_map/tuning_map[0]))

def gpv_roll_recursive (breed, subgroup = None, n = 1):
    """
    Enter a GPV, finds the n-th next GPV. n can be any integer. 
    The subgroup basis must be normalized to all-positive pitches. 
    Doesn't handle some nontrivial subgroups. 
    """
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    just_tuning_map = subgroup.just_tuning_map ()
    if not __is_gpv (breed, just_tuning_map): #verify input
        raise ValueError ("input is not a GPV. ")
    if not isinstance (n, (int, np.integer)):
        raise TypeError ("n must be an integer. ")
    return __gpv_roll_recursive (breed, just_tuning_map, n)

def __gpv_roll_recursive (breed, tuning_map, n = 1):
    if n == 0:
        return breed
    else:
        u = 1 if n > 0 else -1
        for i in range (1, len (tuning_map) + 1):
            breed_copy = breed.copy ()
            breed_copy[-i] += u
            if __is_gpv (breed_copy, tuning_map):
                return __gpv_roll_recursive (breed_copy, tuning_map, n - u)
        else:
            raise NotImplementedError ("this nontrivial tuning map cannot be processed. ")

def __nt (n, eq, tuning_map):
    """Finds the tuning map in terms of n-ed-p steps."""
    return n*tuning_map/np.log2 (eq)

def breed2warts (breed, subgroup = None):
    """
    Enter a breed, finds its wart notation. 
    Zero-step equave is disallowed. 
    """
    breed, subgroup = te.setup (breed, subgroup, axis = te.AXIS.VEC)
    if breed[0] == 0:
        raise ValueError ("wart is undefined. ")

    eq = subgroup.ratios ()[0].value ()
    if eq == 2: #octave equave
        prefix = ""
    elif eq in WARTS_DICT: #non-octave wartable equave
        prefix = WARTS_DICT[eq]
    else: #unwartable equave
        prefix = "*"

    just_tuning_map = subgroup.just_tuning_map ()
    if __is_pv (breed, just_tuning_map): #patent val
        postfix = ""
    elif all (entry in WARTS_DICT for entry in subgroup.ratios (evaluate = True)): #nonpatent val in a wartable subgroup
        # find the just tuning map in n-ed-p steps
        # and the corresponding patent val
        nj = __nt (breed[0], eq, just_tuning_map)
        pv = np.rint (nj)

        # find the number of wart letters for each prime
        # which equals twice the difference between the values in our breed and the pv
        # if it deviates from the pv value on the same side as the pv value deviates from just
        # or that minus 1 otherwise
        warts_number_list = (
            2*np.fabs (breed - pv).astype (int) + np.where ((breed - pv)*(pv - nj) >= 0., 0, -1))

        postfix = ""
        for i, si in enumerate (subgroup.ratios (evaluate = True)):
            postfix += warts_number_list[i]*str (WARTS_DICT[si])
    else: #nonpatent val in an unwartable subgroup
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
        return np.rint (__nt (warts, 2, subgroup.just_tuning_map ())).astype (int)
    else:
        raise TypeError ("Enter a string or number. ")

def __warts2breed (warts, subgroup):
    match = re.match (r"^([a-x]?)(\d+)([a-x]*)", str (warts))
    if not match or (match.group (1) and re.match (match.group (1), match.group (3))):
        raise ValueError ("Invalid wart notation. ")

    # find the equave from the wart prefix
    if match.group (1):
        for i, wi in WARTS_DICT.items ():
            if wi == match.group (1):
                wart_eq = i
                break
    else:
        wart_eq = 2

    # find the number of each wart letter
    warts_number_list = np.zeros (len (subgroup))
    for i, si in enumerate (subgroup.ratios (evaluate = True)):
        if si in WARTS_DICT: # wart is supported
            warts_number_list[i] = len (re.findall (WARTS_DICT[si], match.group (3)))

    # find the just tuning map in n-ed-p steps
    # and the corresponding patent val
    nj = __nt (int (match.group (2)), wart_eq, subgroup.just_tuning_map ())
    pv = np.rint (nj)
    
    # find the breed to add to the patent val
    # each entry equals half the number of wart letters for that prime
    # but add 1 before halving if the number of wart letters is odd
    # the sign is the same as how the pv value deviates from just
    # if the number of wart letters is even
    # or opposite otherwise
    alt_breed = np.copysign (
        np.ceil (warts_number_list/2), np.where (warts_number_list % 2 == 0, 1, -1)*(pv - nj))

    return (pv + alt_breed).astype (int)

def is_monotonic (breed, monzos, show = False):
    """
    Enter a breed and a monzo matrix. 
    Returns whether the breed is monotonic
    with respect to the intervals represented by the monzos. 
    """
    subgroup = te.Subgroup (monzos = np.eye (monzos.shape[0], dtype = int))
    just_tuning_map = subgroup.just_tuning_map ()
    return __is_monotonic (breed, monzos, just_tuning_map, show = show)

def __is_monotonic (breed, monzos, tuning_map, show = False):
    monzos = monzos[:, np.argsort (tuning_map @ monzos)]
    step_spectrum = np.squeeze (breed @ monzos)
    if show: 
        for i, si in enumerate (step_spectrum):
            ratio = te.monzo2ratio (monzos.T[i])
            print (f"{ratio}", si)
    if len (step_spectrum) == 1:
        return True 
    else: 
        return all (step_spectrum[k + 1] >= step_spectrum[k] 
            for k in range (len (step_spectrum) - 1))
