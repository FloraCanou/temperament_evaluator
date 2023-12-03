# © 2020-2023 Flora Canou | Version 0.27.0
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

def vec_pad (vec, length):
    """Pads a vector with zeros to a specified length."""
    vec_copy = np.array (vec)
    vec_copy.resize (length)
    return vec_copy

def column_stack_pad (vec_list, length = None):
    """Column-stack with zero-padding."""
    length = length or max (vec_list, key = len).__len__ () # finds the max length
    return np.column_stack ([vec_pad (vec, length) for vec in vec_list])

class Norm: 
    """Norm profile for the tuning space."""

    def __init__ (self, wtype = "tenney", wamount = 1, skew = 0, order = 2):
        self.wtype = wtype
        self.wamount = wamount
        self.skew = skew
        self.order = order

    def __get_interval_weight (self, primes):
        """Returns the weight matrix for a list of formal primes. """
        match self.wtype:
            case "tenney":
                weight_vec = np.log2 (primes)
            case "wilson" | "benedetti":
                weight_vec = np.asarray (primes)
            case "equilateral":
                weight_vec = np.ones (len (primes))
            # case "hahn24": #pending better implementation
            #     weight_vec = np.ceil (np.log2 (primes)/np.log2 (24))
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                self.wtype = "tenney"
                return self.__get_weight (primes)
        return np.diag (weight_vec**self.wamount)

    def __get_tuning_weight (self, primes):
        return linalg.inv (self.__get_interval_weight (primes))

    def __get_interval_skew (self, primes):
        """Returns the skew matrix for a list of formal primes. """
        if self.skew == 0:
            return np.eye (len (primes))
        elif self.order == 2:
            return np.append (np.eye (len (primes)), self.skew*np.ones ((1, len (primes))), axis = 0)
        else:
            raise NotImplementedError ("Skew only works with Euclidean norm as of now.")

    def __get_tuning_skew (self, primes):
        # return linalg.pinv (self.__get_interval_skew (primes)) # same but for skew = np.inf
        if self.skew == 0:
            return np.eye (len (primes))
        elif self.order == 2:
            r = 1/(len (primes)*self.skew + 1/self.skew)
            kr = 1/(len (primes) + 1/self.skew**2)
            return np.append (
                np.eye (len (primes)) - kr*np.ones ((len (primes), len (primes))),
                r*np.ones ((len (primes), 1)), 
                axis = 1
            )
        else:
            raise NotImplementedError ("Skew only works with Euclidean norm as of now.")

    def tuning_x (self, main, subgroup):
        return main @ self.__get_tuning_weight (subgroup) @ self.__get_tuning_skew (subgroup)

    def interval_x (self, main, subgroup):
        return self.__get_interval_skew (subgroup) @ self.__get_interval_weight (subgroup) @ main

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

def __get_length (main, axis):
    """Gets the length along a certain axis."""
    match axis:
        case AXIS.ROW:
            return main.shape[1]
        case AXIS.COL:
            return main.shape[0]
        case AXIS.VEC:
            return main.size

def get_subgroup (main, axis):
    """Gets the default subgroup along a certain axis."""
    return PRIME_LIST[:__get_length (main, axis)]

def setup (main, subgroup, axis):
    """Tries to match the dimensions along a certain axis."""
    main = np.asarray (main)
    if subgroup is None:
        subgroup = get_subgroup (main, axis)
    elif (length_main := __get_length (main, axis)) != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (length_main, len (subgroup))
        match axis:
            case AXIS.ROW:
                main = main[:, :dim]
            case AXIS.COL:
                main = main[:dim, :]
            case AXIS.VEC:
                main = main[:dim]
        subgroup = subgroup[:dim]
    return main, subgroup

def monzo2ratio (monzo, subgroup = None):
    """
    Takes a monzo, returns the ratio in [num, den] form, 
    subgroup monzo supported.
    ratio[0]: num, ratio[1]: den
    """
    monzo, subgroup = setup (monzo, subgroup, axis = AXIS.VEC)
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
        if ratio[0] == 1 and ratio[1] == 1:
            break
    else:
        raise ValueError ("improper subgroup. ")

    return np.trim_zeros (np.array (monzo), trim = "b") if trim else np.array (monzo)

def bra (covector):
    return "<" + " ".join (map (str, np.trim_zeros (covector, trim = "b"))) + "]"

def ket (vector):
    return "[" + " ".join (map (str, np.trim_zeros (vector, trim = "b"))) + ">"

def matrix2array (main):
    """Takes a possibly fractional sympy matrix and converts it to an integer numpy array."""
    return np.array (main/functools.reduce (gcd, tuple (main)), dtype = int).squeeze ()

def nullspace (covectors):
    frac_nullspace_matrix = Matrix (covectors).nullspace ()
    return np.column_stack ([matrix2array (entry) for entry in frac_nullspace_matrix])

def antinullspace (vectors):
    frac_antinullspace_matrix = Matrix (np.flip (vectors.T)).nullspace ()
    return np.flip (np.row_stack ([matrix2array (entry) for entry in frac_antinullspace_matrix]))

def show_monzo_list (monzos, subgroup):
    """
    Takes an array of monzos and show them in a readable manner. 
    Used to display comma bases and eigenmonzo bases. 
    """
    for entry in monzos.T:
        monzo_str = ket (entry)
        if np.log2 (subgroup) @ np.abs (entry) < 53: # shows the ratio for those < ~1e16
            ratio = monzo2ratio (entry, subgroup)
            print (monzo_str, f"({ratio[0]}/{ratio[1]})")
        else:
            print (monzo_str)
