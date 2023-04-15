# © 2020-2023 Flora Canou | Version 0.25.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, normalforms
from sympy import gcd

ROW, COL, VEC = 0, 1, 2
PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
SCALAR = 1200 #could be in octave, but for precision reason
RATIONAL_WEIGHT_LIST = ["equilateral"]
ALGEBRAIC_WEIGHT_LIST = RATIONAL_WEIGHT_LIST + ["wilson", "benedetti"]

def as_list (a):
    return a if isinstance (a, list) else [a]

# norm profile for the tuning space
class Norm: 
    def __init__ (self, wtype = "tenney", wamount = 1, skew = 0, order = 2):
        self.wtype = wtype
        self.wamount = wamount
        self.skew = skew
        self.order = order

# normalizes the matrix to HNF
def __hnf (main, mode = ROW):
    if mode == ROW:
        return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (main)).T).T, dtype = int))
    elif mode == COL:
        return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (main))), dtype = int))

# saturates the matrix, pernet--stein method
def __sat (main):
    r = Matrix (main).rank ()
    return np.rint (
        linalg.inv (__hnf (main, mode = COL)[:, :r]) @ main
        ).astype (int)

# saturation & normalization
# normalization only checks multirank matrices
def canonicalize (main, saturate = True, normalize = True, axis = ROW):
    if axis == ROW:
        main = __sat (main) if saturate else main
        main = __hnf (main) if normalize and main.shape[0] > 1 else main
    elif axis == COL:
        main = np.flip (__sat (np.flip (main).T)).T if saturate else main
        main = np.flip (__hnf (np.flip (main).T)).T if normalize and main.shape[1] > 1 else main
    return main

canonicalise = canonicalize

# gets the subgroup and tries to match the dimensions
def get_subgroup (main, subgroup, axis):
    main = np.asarray (main)
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

def __get_weight (subgroup, wtype = "tenney", wamount = 1):
    if wtype == "tenney":
        weight_vec = np.reciprocal (np.log2 (np.array (subgroup, dtype = float)))
    elif wtype == "wilson" or wtype == "benedetti":
        weight_vec = np.reciprocal (np.array (subgroup, dtype = float))
    elif wtype == "equilateral":
        weight_vec = np.ones (len (subgroup))
    else:
        warnings.warn ("weighter type not supported, using default (\"tenney\")")
        return get_weight (subgroup, wtype = "tenney", wamount = wamount)
    return np.diag (weight_vec**wamount)

def __get_skew (subgroup, skew = 0, order = 2):
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

def weightskewed (main, subgroup, norm):
    return (main
        @ __get_weight (subgroup, norm.wtype, norm.wamount) 
        @ __get_skew (subgroup, norm.skew, norm.order))

# takes a monzo, returns the ratio in [num, den] form
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
    if (not all (isinstance (entry, (int, np.integer)) for entry in ratio)
        or any (entry < 1 for entry in ratio)):
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

    return np.trim_zeros (np.array (monzo), trim = "b") if trim else np.array (monzo)

def bra (val):
    return "<" + " ".join (map (str, np.trim_zeros (val, trim = "b"))) + "]"

def ket (monzo):
    return "[" + " ".join (map (str, np.trim_zeros (monzo, trim = "b"))) + ">"

# takes a possibly fractional sympy matrix and converts it to an integer numpy array
def matrix2array (main):
    return np.array (main/gcd (tuple (main)), dtype = int).squeeze ()

# takes a list (python list) of monzos (sympy matrices) and show them in a readable manner
# used for comma basis and eigenmonzo basis
def show_monzo_list (monzo_list, subgroup):
    for entry in monzo_list:
        monzo = matrix2array (entry)
        monzo_str = ket (monzo)
        if np.log2 (subgroup) @ np.abs (monzo) < 53: # shows the ratio for those < ~1e16
            ratio = monzo2ratio (monzo, subgroup)
            print (monzo_str, f"({ratio[0]}/{ratio[1]})")
        else:
            print (monzo_str)
