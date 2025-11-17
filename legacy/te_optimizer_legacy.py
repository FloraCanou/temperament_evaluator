# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.
# Version 0.30.0

import warnings
import numpy as np
from scipy import optimize, linalg
np.set_printoptions (suppress = True, linewidth = 256, precision = 3)

PRIME_LIST = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 
    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]

class SCALAR:
    CENT = 1200

class Norm: 
    """Norm profile for the tuning space."""

    def __init__ (self, wtype = None, wmode = 1, wstrength = 1, skew = 0, order = 2):
        if wtype: 
            wmode, wstrength = self.__presets (wtype)
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

        def modal_weighter (primes, m): 
            if m == 0: 
                return primes
            elif m > 0: 
                return modal_weighter (2*np.log2 (primes), m - 1)
            else: 
                return modal_weighter (np.exp2 (primes/2), m + 1)

        return (modal_weighter (np.asarray (primes), self.wmode)/2)**self.wstrength

    def val_weight (self, primes):
        """Returns the val weight matrix for a list of formal primes. """
        return np.diag (1/self.__weight_vec (primes))

    def val_skew (self, subgroup):
        """Returns the val skew matrix for a list of formal primes. """
        if self.skew == 0:
            return np.eye (len (subgroup))
        elif self.order == 2:
            r = 1/(len (subgroup)*self.skew + 1/self.skew)
            kr = 1/(len (subgroup) + 1/self.skew**2)
        else:
            raise NotImplementedError ("Weil skew only works with Euclidean norm as of now.")
        return np.append (
            np.eye (len (subgroup)) - kr*np.ones ((len (subgroup), len (subgroup))),
            r*np.ones ((len (subgroup), 1)), axis = 1)

    def val_transform (self, main, subgroup):
        return main @ self.val_weight (subgroup) @ self.val_skew (subgroup)

def __get_subgroup (main, subgroup):
    main = np.asarray (main)
    if subgroup is None:
        subgroup = PRIME_LIST[:main.shape[1]]
    elif main.shape[1] != len (subgroup):
        warnings.warn ("dimension does not match. Casting to the smaller dimension. ")
        dim = min (main.shape[1], len (subgroup))
        main = main[:, :dim]
        subgroup = subgroup[:dim]
    return main, subgroup

def optimizer_main (breeds, subgroup = None, norm = Norm (), 
        cons_monzo_list = None, des_monzo = None, show = True):
    # NOTE: "map" is a reserved word
    # optimization is preferably done in the unit of octaves, but for precision reasons
    breeds, subgroup = __get_subgroup (breeds, subgroup)

    just_tuning_map = SCALAR.CENT*np.log2 (subgroup)
    breeds_x = norm.val_transform (breeds, subgroup)
    just_tuning_map_x = norm.val_transform (just_tuning_map, subgroup)
    if norm.order == 2 and cons_monzo_list is None: #simply using lstsq for better performance
        res = linalg.lstsq (breeds_x.T, just_tuning_map_x)
        gen = res[0]
        if show: 
            print ("Euclidean tuning without constraints, solved using lstsq. ")
    else:
        gen0 = just_tuning_map[:breeds.shape[0]] #initial guess
        if cons_monzo_list is None:
            cons_object = ()
        else:
            cons_object = optimize.LinearConstraint ((breeds @ cons_monzo_list).T, 
                lb = (just_tuning_map @ cons_monzo_list).T, 
                ub = (just_tuning_map @ cons_monzo_list).T)
        res = optimize.minimize (
            lambda gen: linalg.norm (gen @ breeds_x - just_tuning_map_x, ord = norm.order), 
            gen0, method = "COBYQA", constraints = cons)
        if show:
            print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if des_monzo is not None:
        if np.asarray (des_monzo).ndim > 1 and np.asarray (des_monzo).shape[1] != 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (des_tempered_size := gen @ breeds @ des_monzo) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            gen *= (just_tuning_map @ des_monzo)/des_tempered_size

    tempered_tuning_map = gen @ breeds
    error_map = tempered_tuning_map - just_tuning_map

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")

    return gen, tempered_tuning_map, error_map

optimiser_main = optimizer_main
