# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.
# Version 1.11.1

import re, functools, itertools, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, normalforms
from sympy import gcd
np.set_printoptions (suppress = True, linewidth = 256, precision = 3)

PRIME_LIST = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 
    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 
]
RATIONAL_WEIGHT_LIST = ["equilateral"]
ALGEBRAIC_WEIGHT_LIST = RATIONAL_WEIGHT_LIST + ["wilson", "benedetti"]

class AXIS:
    ROW, COL, VEC = 0, 1, 2

class SCALAR:
    OCTAVE = 1
    MILLIOCTAVE = 1000
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
        if not self.__is_regular (num, den): 
            raise ValueError ("irregular number not supported yet. ")
        self.pos, self.num, self.den = self.__reduce (num, den)

    @staticmethod
    def __is_regular (num, den):
        """Checks whether the ratio is nonzero and finite."""
        return all (np.isfinite ((num, den))) and all ((num, den))

    @staticmethod
    def __reduce (num, den):
        """Determines the sign and eliminates the common factors."""
        pos = (num > 0) == (den > 0)
        if isinstance (num, (int, np.integer)) and isinstance (den, (int, np.integer)):
            gcd = np.gcd (num, den)
            num = abs (num//gcd)
            den = abs (den//gcd)
        else:
            warnings.warn ("non-integer input. ")
            num = abs (np.divide (num, den))
            den = 1
        return pos, num, den

    @classmethod
    def from_string (cls, s):
        """Creates a ratio from a string."""
        match = re.match (r"^(-?)(\d*)\/?(\d*)$", s)
        num = int (match.group (2) or "1")
        den = int (match.group (3) or "1")
        if match.group (1): num = -num
        return cls (num, den)
    
    @classmethod
    def from_tuple (cls, t):
        """Creates a ratio from a list/tuple."""
        match len (t):
            case 0:
                return cls (1, 1)
            case 1:
                return cls (t[0], 1)
            case 2: 
                return cls (t[0], t[1])
            case _:
                raise IndexError ("too many values provided.")

    def value (self):
        """Returns the value of the ratio as a float."""
        v = np.divide (self.num, self.den)
        return v if self.pos else -v

    def oct_count (self): 
        """Returns the number of full octaves the ratio has."""
        # NOTE: "oct" is a reserved word
        return np.floor (np.log2 (self.value ())).astype (int)

    def eq_count (self, eq): 
        """
        Enter a ratio for the equave, 
        returns the number of full equaves the ratio has.
        """
        eq = as_ratio (eq)
        if eq == 2.: 
            return self.oct_count ()
        else:
            return self.__eq_count (eq)
            
    def __eq_count (self, eq):
        return (np.log2 (self.value ())//np.log2 (eq.value ())).astype (int)

    def oct_reduce (self): 
        """Returns the octave-reduced ratio."""
        # NOTE: "oct" is a reserved word
        oct_count = self.oct_count ()
        if oct_count == 0:
            return self
        elif oct_count > 0:
            return Ratio (self.num, self.den*2**oct_count)
        else:
            return Ratio (self.num*2**(-oct_count), self.den)

    def octave_reduce (self):
        """Same as above. Deprecated since 1.11.0. """
        warnings.warn ("`octave_reduce` has been deprecated. "
            "Use `oct_reduce` instead. ", FutureWarning)
        return self.oct_reduce ()

    def eq_reduce (self, eq):
        """Enter a ratio for the equave, returns the equave-reduced ratio."""
        eq = as_ratio (eq)
        if eq == 2.:
            return self.oct_reduce ()
        else:
            return self.__eq_reduce (eq)

    def __eq_reduce (self, eq):
        eq_count = self.__eq_count (eq)
        if eq_count == 0:
            return self
        elif eq_count > 0:
            return Ratio (self.num*eq.den**eq_count, self.den*eq.num**eq_count)
        else:
            return Ratio (self.num*eq.num**(-eq_count), self.den*eq.den**(-eq_count))

    def oct_complement (self):
        """Returns the octave-complement ratio."""
        # NOTE: "oct" is a reserved word
        return Ratio (self.den*2, self.num)

    def eq_complement (self, eq):
        """Enter a ratio for the equave, returns the equave-complement ratio."""
        eq = as_ratio (eq)
        if eq == 2.:
            return self.oct_complement ()
        else:
            return self.__eq_complement (eq)

    def __eq_complement (self, eq):    
        return Ratio (self.den*eq.num, self.num*eq.den)

    def __str__ (self):
        s = f"{self.num}" if self.den == 1 else f"{self.num}/{self.den}"
        return s if self.pos else "-" + s

    def __eq__ (self, other):
        return self.value () == (other.value () if isinstance (other, Ratio) else other)

def as_ratio (n):
    """Returns a ratio object, fractional notation supported."""
    if isinstance (n, Ratio):
        return n
    elif isinstance (n, str):
        return Ratio.from_string (n)
    elif isinstance (n, (list, tuple)):
        return Ratio.from_tuple (n)
    elif np.asarray (n).size == 1: 
        return Ratio (n, 1)
    else:
        raise TypeError ("unsupported type.")

class Subgroup:
    """Subgroup profile of ji."""

    def __init__ (self, ratios = None, monzos = None, saturate = False, normalize = True):
        if (ratios is None) == (monzos is None): 
            raise ValueError ("Either ratios or monzos must be provided.")
        
        # construct the basis matrix
        self.basis_matrix = canonicalize (
                monzos or column_stack_pad (
                [ratio2monzo (as_ratio (entry)) for entry in ratios]
                ), saturate, normalize, axis = AXIS.COL)

        # normalize to positive pitches
        for i, si in enumerate (self.basis_matrix.T):
            ratio = monzo2ratio (si)
            if ratio.num < ratio.den: 
                self.basis_matrix[:, i] *= -1

    def basis_matrix_to (self, other):
        """
        Returns the basis matrix with respect to another subgroup.
        Also useful for padding zeros.
        """

        # pad zeros
        max_length = max (self.basis_matrix.shape[0], other.basis_matrix.shape[0])
        self_basis_matrix = column_stack_pad (self.basis_matrix.T, length = max_length)
        other_basis_matrix = column_stack_pad (other.basis_matrix.T, length = max_length)

        # find the subgroup basis matrix
        result = linalg.pinv (other_basis_matrix) @ self_basis_matrix

        # test for disjoint or degenerate subgroup
        # NOTE: linalg.pinv can introduce small numerical errors
        if not np.allclose (other_basis_matrix @ result, self_basis_matrix, rtol = 0, atol = 1e-6): 
            raise ValueError ("disjoint or degenerate subgroup. ")
        
        # convert to integer type if possible
        result_rd = np.rint (result)
        if np.allclose (result, result_rd, rtol = 0, atol = 1e-6):
            result = result_rd.astype (int)
        else:
            warnings.warn ("non-integer subgroup basis. Possibly improper subgroup. ")

        return result
    
    def ratios (self, evaluate = False):
        """Returns a list of ratio objects or floats."""
        if evaluate:
            return [monzo2ratio (entry).value () for entry in self.basis_matrix.T]
        else:
            return [monzo2ratio (entry) for entry in self.basis_matrix.T]

    def just_tuning_map (self, scalar = SCALAR.OCTAVE): #in octaves by default
        return scalar*np.log2 (self.ratios (evaluate = True))

    def is_prime (self):
        """
        Returns whether the subgroup has a basis of only primes.
        Such a basis allows no distinction between inharmonic and subgroup tunings
        for all norms. 
        """
        ratios = self.ratios (evaluate = True)
        return all (entry in PRIME_LIST for entry in ratios)

    def is_prime_power (self):
        """
        Returns whether the subgroup has a basis of only primes and/or their powers.
        Such a basis allows no distinction between inharmonic and subgroup tunings
        for tenney-weighted norms. 
        """
        return (all (np.count_nonzero (row) <= 1 for row in self.basis_matrix)
            and all (np.count_nonzero (row) <= 1 for row in self.basis_matrix.T))

    def minimal_prime_subgroup (self):
        """Returns the smallest prime subgroup that contains this subgroup. """
        group = PRIME_LIST[:self.basis_matrix.shape[0]]
        selector = self.basis_matrix.any (axis = 1)
        return Subgroup (ratios = list (itertools.compress (group, selector)))

    def index (self):
        """
        Returns what fraction the subgroup is to its minimal prime subgroup. 
        1: it's a prime subgroup. 
        inf: it's a degenerate subgroup. 
        Temperament complexity can be defined on any nondegenerate subgroups. 
        """
        try:
            return linalg.det (self.basis_matrix_to (self.minimal_prime_subgroup ()))
        except ValueError:
            return np.inf

    def __str__ (self):
        return ".".join (entry.__str__ () for entry in self.ratios ())

    def __len__ (self):
        """Returns its own rank."""
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
            #     weight_vec = np.reciprocal (np.floor (np.log2 (24)/np.log2 (primes)))
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                self.wtype = "tenney"
                return self.__get_interval_weight (primes)
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

# canonicalization functions

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

# initialization functions

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

# conversion functions

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
    primes = PRIME_LIST[:len (monzo)]
    num, den = 1, 1
    for i, mi in enumerate (monzo):
        if mi > 0:
            num *= primes[i]**mi
        elif mi < 0:
            den *= primes[i]**(-mi)
    return Ratio (num, den)

def ratio2monzo (ratio, subgroup = None):
    """
    Takes a ratio object, returns the monzo, 
    subgroup monzo supported.
    """
    if ratio.value () < 0:
        raise ValueError ("invalid ratio. ")

    monzo = __ratio2monzo (ratio)
    if subgroup is None:
        return monzo
    else:
        # pad zeros
        max_length = max (len (monzo), subgroup.basis_matrix.shape[0])
        monzo = vec_pad (monzo, length = max_length)
        subgroup_basis_matrix = column_stack_pad (subgroup.basis_matrix.T, length = max_length)

        # find the subgroup monzo
        subgroup_monzo = linalg.pinv (subgroup_basis_matrix) @ monzo

        # test for disjoint or degenerate subgroup
        # NOTE: linalg.pinv can introduce small numerical errors
        if not np.allclose (subgroup_basis_matrix @ subgroup_monzo, monzo, rtol = 0, atol = 1e-6): 
            raise ValueError ("disjoint or degenerate subgroup. ")
        
        # convert to integer type if possible
        subgroup_monzo_rd = np.rint (subgroup_monzo)
        if np.allclose (subgroup_monzo, subgroup_monzo_rd, rtol = 0, atol = 1e-6):
            subgroup_monzo = subgroup_monzo_rd.astype (int)
        else:
            warnings.warn ("non-integer subgroup monzo. Possibly improper subgroup. ")
        
        return subgroup_monzo

def __ratio2monzo (ratio):
    monzo = []
    primes = PRIME_LIST
    for entry in primes:
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

def matrix2array (main):
    """Takes a possibly fractional sympy matrix and converts it to an integer numpy array."""
    return np.array (main/functools.reduce (gcd, tuple (main)), dtype = int).squeeze ()

def nullspace (covectors):
    """Row-style nullspace."""
    frac_nullspace_matrix = Matrix (covectors).nullspace ()
    return np.column_stack ([matrix2array (entry) for entry in frac_nullspace_matrix])

def antinullspace (vectors):
    """
    Column-style nullspace. 
    *Antinullspace* is a term that's supposed to be eliminated.
    It's antitranspose--nullspace--antitranspose 
    where *antitranspose* refers to flip and transpose. 
    """
    frac_antinullspace_matrix = Matrix (np.flip (vectors.T)).nullspace ()
    return np.flip (np.row_stack ([matrix2array (entry) for entry in frac_antinullspace_matrix]))

# annotation functions

def bra (covector):
    return "<" + " ".join (map (str, np.trim_zeros (covector, trim = "b"))) + "]"

def ket (vector):
    return "[" + " ".join (map (str, np.trim_zeros (vector, trim = "b"))) + ">"

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
