# © 2020-2022 Flora Canou | Version 0.22.2
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from sympy.matrices import Matrix, normalforms
from sympy import gcd

ROW, COL, VEC = 0, 1, 2
PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SCALAR = 1200 #could be in octave, but for precision reason
RATIONAL_WEIGHT_LIST = ["equilateral", "frobenius"]
ALGEBRAIC_WEIGHT_LIST = RATIONAL_WEIGHT_LIST + ["wilson", "benedetti"]

def as_list (a):
    return a if isinstance (a, list) else [a]

# normalizes matrices to HNF, only checks multirank matrices
def hnf (main):
    return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (main)).T).T).astype (int))

def normalize (main, axis = ROW):
    if axis == ROW:
        return hnf (main) if main.shape[0] > 1 else main
    elif axis == COL:
        return np.flip (hnf (np.flip (main).T)).T if main.shape[1] > 1 else main

# gets the subgroup and tries to match the dimensions
def get_subgroup (main, subgroup, axis):
    if axis == ROW:
        length_main = main.shape[1]
    elif axis == COL:
        length_main = main.shape[0]
    elif axis == VEC:
        length_main = len (main)

    if subgroup is None:
        subgroup = PRIME_LIST[:length_main]
    elif length_main != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (length_main, len (subgroup))
        if axis == ROW:
            main = main[:, :dim]
        elif axis == COL:
            main = main[:dim, :]
        elif axis == VEC:
            main = main[:dim]
        subgroup = subgroup[:dim]

    return main, subgroup

def get_weight (subgroup, wtype = "tenney", wamount = 1):
    if wtype == "tenney":
        weight_vec = np.reciprocal (np.log2 (np.array (subgroup, dtype = float)))
    elif wtype == "wilson" or wtype == "benedetti":
        weight_vec = np.reciprocal (np.array (subgroup, dtype = float))
    elif wtype == "equilateral":
        weight_vec = np.ones (len (subgroup))
    elif wtype == "frobenius":
        warnings.warn ("\"frobenius\" is deprecated. Use \"equilateral\" instead. ")
        weight_vec = np.ones (len (subgroup))
    else:
        warnings.warn ("weighter type not supported, using default (\"tenney\")")
        return get_weight (subgroup, wtype = "tenney", wamount = wamount)
    return np.diag (weight_vec**wamount)

def get_skew (subgroup, skew = 0, order = 2):
    if skew == 0:
        return np.eye (len (subgroup))
    elif order == 2:
        if not skew == np.inf:
            r = skew/(len (subgroup)*skew**2 + 1)
            kr = skew*r
        else:
            r = 0
            kr = 1/len (subgroup)
    else:
        raise NotImplementedError ("Skew only works with Euclidean norm as of now.")
    return np.append (
        np.eye (len (subgroup)) - kr*np.ones ((len (subgroup), len (subgroup))),
        r*np.ones ((len (subgroup), 1)), axis = 1)

def weightskewed (main, subgroup, wtype = "tenney", wamount = 1, skew = 0, order = 2):
    return main @ get_weight (subgroup, wtype, wamount) @ get_skew (subgroup, skew, order)

# takes a monzo, returns the ratio in [num, den] form
# doesn't validate the basis
# ratio[0]: num, ratio[1]: den
def monzo2ratio (monzo, subgroup = None):
    monzo, subgroup = get_subgroup (monzo, subgroup, axis = VEC)
    ratio = [1, 1]
    for i, mi in enumerate (monzo):
        if mi > 0:
            ratio[0] *= subgroup[i]**mi
        elif mi < 0:
            ratio[1] *= subgroup[i]**(-mi)
    return ratio

# takes a ratio in [num, den] form, returns the monzo
def ratio2monzo (ratio, subgroup = None):
    if not all (isinstance (entry, (int, np.integer)) for entry in ratio) or any (entry < 1 for entry in ratio):
        raise ValueError ("numerator and denominator should be positive integers. ")
    if trim := (subgroup is None):
        subgroup = PRIME_LIST

    monzo = [0]*len (subgroup)
    for i, si in enumerate (subgroup):
        while ratio[0] % si == 0:
            monzo[i] += 1
            ratio[0] /= si
        while ratio[1] % si == 0:
            monzo[i] -= 1
            ratio[1] /= si
        if all (entry == 1 for entry in ratio):
            break
    else:
        raise ValueError ("improper subgroup. ")

    return np.array (np.trim_zeros (monzo, trim = "b") if trim else monzo)

# takes a list (python list) of monzos (sympy matrices) and show them in a readable manner
# used for comma basis and eigenmonzo basis
def show_monzo_list (monzo_list, subgroup):
    for entry in monzo_list:
        monzo = np.array (entry/gcd (tuple (entry))).squeeze ()
        monzo_str = "[" + " ".join (map (str, np.trim_zeros (monzo, trim = "b"))) + ">"
        if np.log2 (subgroup) @ np.abs (monzo) < 53: # shows the ratio for those < ~1e16
            ratio = monzo2ratio (monzo, subgroup)
            print (monzo_str, f"({ratio[0]}/{ratio[1]})")
        else:
            print (monzo_str)
