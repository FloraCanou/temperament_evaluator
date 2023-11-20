# © 2020-2023 Flora Canou | Version 0.26.3
# This work is licensed under the GNU General Public License version 3.

import functools, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, normalforms
from sympy import gcd

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
RATIONAL_WEIGHT_LIST = ["equilateral"]
ALGEBRAIC_WEIGHT_LIST = RATIONAL_WEIGHT_LIST + ["wilson", "benedetti"]

class AXIS:
    ROW, COL, VEC = 0, 1, 2

class SCALAR:
    OCTAVE = 1
    CENT = 1200

def as_list (main):
    if isinstance (main, list):
        return main
    else:
        try:
            return list (main)
        except TypeError:
            return [main]

class Norm: 
    """Norm profile for the tuning space."""

    def __init__ (self, wtype = "tenney", wamount = 1, skew = 0, order = 2):
        self.wtype = wtype
        self.wamount = wamount
        self.skew = skew
        self.order = order

    def __get_weight (self, subgroup):
        match self.wtype:
            case "tenney":
                weight_vec = np.reciprocal (np.log2 (np.array (subgroup, dtype = float)))
            case "wilson" | "benedetti":
                weight_vec = np.reciprocal (np.array (subgroup, dtype = float))
            case "equilateral":
                weight_vec = np.ones (len (subgroup))
            # case "hahn24": #pending better implementation
                # weight_vec = np.floor (np.log2 (24)/np.log2 (np.array (subgroup, dtype = float)))
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                self.wtype = "tenney"
                return self.__get_weight (subgroup)
        return np.diag (weight_vec**self.wamount)

    def __get_skew (self, subgroup):
        if self.skew == 0:
            return np.eye (len (subgroup))
        elif self.order == 2:
            r = 1/(len (subgroup)*self.skew + 1/self.skew)
            kr = 1/(len (subgroup) + 1/self.skew**2)
        else:
            raise NotImplementedError ("Skew only works with Euclidean norm as of now.")
        return np.append (
            np.eye (len (subgroup)) - kr*np.ones ((len (subgroup), len (subgroup))),
            r*np.ones ((len (subgroup), 1)), axis = 1)

    def weightskewed (self, main, subgroup):
        return main @ self.__get_weight (subgroup) @ self.__get_skew (subgroup)

def __hnf (main, mode = AXIS.ROW):
    """Normalizes a matrix to HNF."""
    if mode == AXIS.ROW:
        return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (main)).T).T, dtype = int))
    elif mode == AXIS.COL:
        return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (main))), dtype = int))

def __sat (main):
    """Saturates a matrix, pernet--stein method."""
    r = Matrix (main).rank ()
    return np.rint (
        linalg.inv (__hnf (main, mode = AXIS.COL)[:, :r]) @ main
        ).astype (int)

def canonicalize (main, saturate = True, normalize = True, axis = AXIS.ROW):
    """
    Saturation & normalization.
    Normalization only checks multirank matrices.
    """
    if axis == AXIS.ROW:
        main = __sat (main) if saturate else main
        main = __hnf (main) if normalize and main.shape[0] > 1 else main
    elif axis == AXIS.COL:
        main = np.flip (__sat (np.flip (main).T)).T if saturate else main
        main = np.flip (__hnf (np.flip (main).T)).T if normalize and main.shape[1] > 1 else main
    return main

canonicalise = canonicalize

def get_subgroup (main, subgroup, axis):
    """Gets the subgroup and tries to match the dimensions."""
    main = np.asarray (main)
    if axis == AXIS.ROW:
        length_main = main.shape[1]
    elif axis == AXIS.COL:
        length_main = main.shape[0]
    elif axis == AXIS.VEC:
        length_main = len (main)

    if subgroup is None:
        subgroup = PRIME_LIST[:length_main]
    elif length_main != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (length_main, len (subgroup))
        if axis == AXIS.ROW:
            main = main[:, :dim]
        elif axis == AXIS.COL:
            main = main[:dim, :]
        elif axis == AXIS.VEC:
            main = main[:dim]
        subgroup = subgroup[:dim]

    return main, subgroup

def monzo2ratio (monzo, subgroup = None):
    """
    Takes a monzo, returns the ratio in [num, den] form, 
    subgroup monzo supported.
    ratio[0]: num, ratio[1]: den
    """
    monzo, subgroup = get_subgroup (monzo, subgroup, axis = AXIS.VEC)
    ratio = [1, 1]
    for i, mi in enumerate (monzo):
        if mi > 0:
            ratio[0] *= subgroup[i]**mi
        elif mi < 0:
            ratio[1] *= subgroup[i]**(-mi)
    return ratio

def ratio2monzo (ratio, subgroup = None):
    """
    Takes a ratio in [num, den] form, returns the monzo, 
    subgroup monzo supported.
    """
    if (not isinstance (ratio[0], (int, np.integer)) or not isinstance (ratio[1], (int, np.integer))
        or ratio[0] < 1 or ratio[1] < 1):
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

def matrix2array (main):
    """Takes a possibly fractional sympy matrix and converts it to an integer numpy array."""
    return np.array (main/functools.reduce (gcd, tuple (main)), dtype = int).squeeze ()

def show_monzo_list (monzo_list, subgroup):
    """
    Takes a list (python list) of monzos (sympy matrices) and show them in a readable manner. 
    Used to display comma bases and eigenmonzo bases. 
    """
    for entry in monzo_list:
        monzo = matrix2array (entry)
        monzo_str = ket (monzo)
        if np.log2 (subgroup) @ np.abs (monzo) < 53: # shows the ratio for those < ~1e16
            ratio = monzo2ratio (monzo, subgroup)
            print (monzo_str, f"({ratio[0]}/{ratio[1]})")
        else:
            print (monzo_str)
