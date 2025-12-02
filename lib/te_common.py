# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.
# Version 1.16.0

import re, itertools, functools, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, normalforms
from sympy import gcd
np.set_printoptions (suppress = True, linewidth = 256, precision = 3)

def primes_gen ():
    """
    Prime number generator. Infinite sieve of Erathosthenes from
    https://eli.thegreenplace.net/2023/my-favorite-prime-number-generator/. 
    """
    composites = {}
    q = 2
    while True:
        if q not in composites:
            composites[q**2] = [q]
            yield q
        else:
            for p in composites[q]:
                composites.setdefault(p + q, []).append(p)
            del composites[q]
        q += 1

# stock prime list, up to the 24th (89)
# create on start
PRIME_LIST_LEN = 24
PRIME_LIST = list (itertools.islice (primes_gen (), PRIME_LIST_LEN))

def prime_list (length): 
    """Returns the list of prime numbers of a specified length. """
    if length <= PRIME_LIST_LEN: 
        return PRIME_LIST[:length]
    else: 
        return list (itertools.islice (primes_gen (), length))

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
                monzos if monzos is not None else column_stack_pad (
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
        """Returns the just tuning map. """
        primes = prime_list (self.basis_matrix.shape[0])
        return scalar*(np.log2 (primes) @ self.basis_matrix)

    def is_prime_power (self):
        """
        Returns whether the subgroup has a basis of only primes and/or their powers.
        Such a basis allows no distinction between inharmonic and subgroup tunings
        for tenney-weighted norms. 
        """
        return (all (np.count_nonzero (row) <= 1 for row in self.basis_matrix)
            and all (np.count_nonzero (row) <= 1 for row in self.basis_matrix.T))

    def is_prime (self):
        """
        Returns whether the subgroup has a basis of only primes.
        Such a basis allows no distinction between inharmonic and subgroup tunings
        for all norms. 
        """
        return (self.is_prime_power ()
            and np.all (self.basis_matrix[self.basis_matrix != 0] == 1))

    def minimal_prime_subgroup (self):
        """Returns the smallest prime subgroup that contains this subgroup. """
        selector = self.basis_matrix.nonzero ()[0]
        monzos = np.zeros ((self.basis_matrix.shape[0], len (selector)), dtype = int)
        monzos[selector, np.arange (len (selector))] = 1
        return Subgroup (monzos = monzos)

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
    """Tenney--Wilson parametric norm profile for the tuning space."""

    def __init__ (self, wtype = None, wmode = 1, wstrength = 1, skew = 0, order = 2, 
            *, wamount = None):
        if wtype: 
            wmode, wstrength = self.__presets (wtype)
        
        if wamount:
            warnings.warn ("`wamount` is deprecated. Use `wstrength` instead. ", FutureWarning)
            wstrength = wamount

        self.wmode = wmode
        self.wstrength = wstrength
        self.skew = skew
        self.order = order

    @staticmethod
    def __presets (wtype):
        match wtype: 
            case "tenney":
                wmode, wstrength = 1, 1
            case "wilson" | "benedetti":
                wmode, wstrength = 0, 1
            case "equilateral":
                wmode, wstrength = 0, 0
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                wmode, wstrength = 1, 1
        return wmode, wstrength

    def __weight_vec (self, primes):
        """Returns the interval weight vector for a list of formal primes. """

        if not isinstance (self.wmode, (int, np.integer)):
            raise TypeError ("non-integer modes not supported. ")
        wmode = self.wmode if self.wstrength else 0
        wstrength = self.wstrength

        def modal_weighter (primes, m): 
            if m == 0: 
                return primes
            elif m > 0: 
                return modal_weighter (2*np.log2 (primes), m - 1)
            else: 
                return modal_weighter (np.exp2 (primes/2), m + 1)

        return (modal_weighter (np.asarray (primes), wmode)/2)**wstrength

    def interval_weight (self, primes):
        """Returns the interval weight matrix for a list of formal primes. """
        return np.diag (self.__weight_vec (primes))

    def val_weight (self, primes):
        """Returns the val weight matrix for a list of formal primes. """
        return np.diag (1/self.__weight_vec (primes))

    def interval_skew (self, primes):
        """Returns the interval skew matrix for a list of formal primes. """
        if self.skew == 0:
            return np.eye (len (primes))
        elif self.order == 2:
            return np.append (np.eye (len (primes)), self.skew*np.ones ((1, len (primes))), axis = 0)
        else:
            raise NotImplementedError ("Skew only works with Euclidean norm as of now.")

    def val_skew (self, primes):
        """Returns the val skew matrix for a list of formal primes. """
        if self.skew == 0:
            return np.eye (len (primes))
        elif self.order == 2:
            # return linalg.pinv (self.__get_interval_skew (primes))
            # same but for skew = np.inf and for better performance
            r = 1/(len (primes)*self.skew + 1/self.skew)
            kr = 1/(len (primes) + 1/self.skew**2)
            return np.append (
                np.eye (len (primes)) - kr*np.ones ((len (primes), len (primes))),
                r*np.ones ((len (primes), 1)), axis = 1)
        else:
            raise NotImplementedError ("Skew only works with Euclidean norm as of now.")

    def val_transform (self, vals, subgroup):
        primes = subgroup.ratios (evaluate = True)
        return vals @ self.val_weight (primes) @ self.val_skew (primes)

    def interval_transform (self, intervals, subgroup):
        primes = subgroup.ratios (evaluate = True)
        return self.interval_skew (primes) @ self.interval_weight (primes) @ intervals

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
    """Saturation & normalization."""
    if axis == AXIS.ROW:
        main = __sat (main) if saturate else main
        main = __hnf (main) if normalize else main
    elif axis == AXIS.COL:
        main = np.flip (__sat (np.flip (main).T)).T if saturate else main
        main = np.flip (__hnf (np.flip (main).T)).T if normalize else main
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

def setup (main, subgroup, axis):
    """
    Returns the default subgroup for a main array if the subgroup is not provided. 
    Otherwise, tries to match the dimensionalities for a main matrix and a subgroup
    along a certain axis. 
    """
    main = np.asarray (main)
    if subgroup is None:
        primes = prime_list (__get_length (main, axis))
        subgroup = Subgroup (primes)
    else: 
        length_main = __get_length (main, axis)
        length_subgroup = len (subgroup)
        if length_main != length_subgroup:
            warnings.warn ("dimensionalities do not match. Casting to the smaller dimensionality. ")
            dim = min (length_main, length_subgroup)
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

def monzo2ratio (subgroup_monzo, subgroup = None):
    """
    Takes a monzo, returns the ratio object, 
    subgroup monzo supported.
    """
    if subgroup is None: 
        monzo = subgroup_monzo
    else: 
        monzo = subgroup.basis_matrix @ vec_pad (subgroup_monzo, length = len (subgroup))
    return __monzo2ratio (monzo)

def __monzo2ratio (monzo):
    primes = prime_list (len (monzo))
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
        subgroup_monzo = monzo
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

def __ratio2monzo (ratio, *, primes = PRIME_LIST):
    # optimized for stock primes
    # this is faster than using the generator from the start
    # only switch to the generator on failure
    num, den = ratio.num, ratio.den
    monzo = []
    for entry in primes:
        order = 0
        while num % entry == 0:
            order += 1
            num //= entry
        while den % entry == 0:
            order -= 1
            den //= entry
        monzo.append (order)
        if num == 1 and den == 1:
            break
    else: #retry using the generator
        monzo = __ratio2monzo (ratio, primes = primes_gen ())
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
    Antitranspose--nullspace--antitranspose 
    where *antitranspose* refers to flip and transpose. 
    """
    frac_antinullspace_matrix = Matrix (np.flip (vectors.T)).nullspace ()
    return np.flip (np.row_stack ([matrix2array (entry) for entry in frac_antinullspace_matrix]))

def breeds2mp (breeds, subgroup):
    """
    Enter an array of breeds and a subgroup, 
    returns the corresponding array of breeds in the minimal prime subgroup, 
    and the minimal prime subgroup itself. 
    """
    subgroup_mp = subgroup.minimal_prime_subgroup ()
    subgroup2mp = subgroup.basis_matrix_to (subgroup_mp)
    breeds_mp = canonicalize ( #NOTE: next line can introduce contorsion in rare cases
        antinullspace (subgroup2mp @ nullspace (breeds)))
    return breeds_mp, subgroup_mp

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
