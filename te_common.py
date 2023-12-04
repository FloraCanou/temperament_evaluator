# © 2020-2023 Flora Canou | Version 1.1.0
# This work is licensed under the GNU General Public License version 3.

import re, functools, warnings
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

class Ratio:
    """Ratio in fraction."""

    def __init__ (self, num, den):
        self.num = num
        self.den = den
        self.__reduce ()

    def __reduce (self):
        if isinstance (self.num, (int, np.integer)) and isinstance (self.den, (int, np.integer)):
            gcd = np.gcd (self.num, self.den)
            self.num = round (self.num/gcd)
            self.den = round (self.den/gcd)
        else:
            self.num = np.divide (self.num, self.den)
            self.den = 1

    def value (self):
        return self.num if self.den == 1 else np.divide (self.num, self.den)

    def octave_reduce (self): #NOTE: "oct" is a reserved word
        """Returns the octave-reduced ratio."""
        num_oct = np.floor (np.log2 (self.value ())).astype (int)
        if num_oct == 0:
            return self
        elif num_oct > 0:
            return Ratio (self.num, self.den*2**num_oct)
        elif num_oct < 0:
            return Ratio (self.num*2**(-num_oct), self.den)

    def __str__ (self):
        return f"{self.num}" if self.den == 1 else f"{self.num}/{self.den}"

    def __eq__ (self, other):
        return self.value () == (other.value () if isinstance (other, Ratio) else other)

def as_ratio (n):
    """Returns a ratio object, fractional notation supported."""
    if isinstance (n, Ratio):
        return n
    elif isinstance (n, str):
        match = re.match ("^(\d*)\/?(\d*)$", n)
        num = match.group (1) or "1"
        den = match.group (2) or "1"
        return Ratio (int (num), int (den))
    elif np.asarray (n).size == 1: 
        return Ratio (n, 1)
    else:
        raise IndexError ("only one value is allowed.")

class Subgroup:
    """Subgroup profile of ji."""

    def __init__ (self, ratios = None, monzos = None, saturate = False, normalize = True):
        if not ratios is None: 
            self.basis_matrix = canonicalize (column_stack_pad (
                [ratio2monzo (as_ratio (entry)) for entry in ratios]
                ), saturate, normalize, axis = AXIS.COL
            )
        elif not monzos is None: 
            self.basis_matrix = canonicalize (monzos, saturate, normalize, axis = AXIS.COL)

    def basis_matrix_to (self, other):
        """
        Returns the basis matrix with respect to another subgroup.
        Also useful for padding zeros.
        """
        result = (linalg.pinv (other.basis_matrix) 
            @ column_stack_pad (self.basis_matrix.T, length = other.basis_matrix.shape[0]))
        if all (entry.is_integer () for entry in result.flat):
            return result.astype (int)
        else:
            warnings.warn ("improper subgroup.")
            return result
    
    def ratios (self, evaluate = False):
        """Returns a list of ratio objects or floats."""
        return [monzo2ratio (entry).value () if evaluate else monzo2ratio (entry) for entry in self.basis_matrix.T]

    def just_tuning_map (self, scalar = SCALAR.OCTAVE): #in octaves by default
        return scalar*np.log2 (self.ratios (evaluate = True))

    def is_trivial (self):
        """
        Returns whether the basis consists of only primes.
        Such a basis allows no distinction between inharmonic and subgroup tunings
        for all norms. 
        """
        ratios = self.ratios (evaluate = True)
        return all (entry in PRIME_LIST for entry in ratios)

    def is_tenney_trivial (self):
        """
        Returns whether the basis consists of only primes and/or their multiples.
        Such a basis allows no distinction between inharmonic and subgroup tunings
        for tenney-weighted norms. 
        """
        return (all (np.count_nonzero (entry) <= 1 for entry in self.basis_matrix)
            and all (np.count_nonzero (entry) <= 1 for entry in self.basis_matrix.T))

    def __str__ (self):
        return ".".join (entry.__str__ () for entry in self.ratios ())

    def __len__ (self):
        """Returns its own dimensionality."""
        return self.basis_matrix.shape[1]

    def __eq__ (self, other):
        return np.array_equal (self.basis_matrix, other.basis_matrix) if isinstance (other, Subgroup) else False

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
        primes = subgroup.ratios (evaluate = True)
        return main @ self.__get_tuning_weight (primes) @ self.__get_tuning_skew (primes)

    def interval_x (self, main, subgroup):
        primes = subgroup.ratios (evaluate = True)
        return self.__get_interval_skew (primes) @ self.__get_interval_weight (primes) @ main

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
        main = __hnf (main) if normalize and np.asarray (main).shape[0] > 1 else main
    elif axis == AXIS.COL:
        main = np.flip (__sat (np.flip (main).T)).T if saturate else main
        main = np.flip (__hnf (np.flip (main).T)).T if normalize and np.asarray (main).shape[1] > 1 else main
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
    return Subgroup (PRIME_LIST[:__get_length (main, axis)])

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
        subgroup.basis_matrix = subgroup.basis_matrix[:, :dim]
    return main, subgroup

def monzo2ratio (monzo, subgroup = None):
    """
    Takes a monzo, returns the ratio object, 
    subgroup monzo supported.
    """
    if subgroup is None: 
        return __monzo2ratio (monzo)
    else: 
        return __monzo2ratio (subgroup.basis_matrix @ vec_pad (monzo, length = len (subgroup)))

def __monzo2ratio (monzo):
    group = PRIME_LIST
    num, den = 1, 1
    for i, mi in enumerate (monzo):
        if mi > 0:
            num *= group[i]**mi
        elif mi < 0:
            den *= group[i]**(-mi)
    return Ratio (num, den)

def ratio2monzo (ratio, subgroup = None):
    """
    Takes a ratio object, returns the monzo, 
    subgroup monzo supported.
    """
    value = ratio.value ()
    if value == np.inf or value == 0 or value == np.nan:
        raise ValueError ("invalid ratio. ")
    elif subgroup is None:
        return __ratio2monzo (ratio)
    else:
        result = (linalg.pinv (subgroup.basis_matrix) 
            @ vec_pad (__ratio2monzo (ratio), length = len (subgroup)))
        if all (entry.is_integer for entry in result):
            return result.astype (int)
        else:
            warnings.warn ("improper subgroup. ")
            return result

def __ratio2monzo (ratio):
    monzo = []
    group = PRIME_LIST
    for entry in group:
        order = 0
        while ratio.num % entry == 0:
            order += 1
            ratio.num /= entry
        while ratio.den % entry == 0:
            order -= 1
            ratio.den /= entry
        monzo.append (order)
        if ratio.num == 1 and ratio.den == 1:
            break
    else:
        raise ValueError ("improper subgroup. ")
    return np.array (monzo)

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
        if subgroup.just_tuning_map () @ np.abs (entry) < 53: # shows the ratio for those < ~1e16
            ratio = monzo2ratio (entry, subgroup)
            print (monzo_str, "(" + ratio.__str__ () + ")")
        else:
            print (monzo_str)
