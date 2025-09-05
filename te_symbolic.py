# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import Rational, floor, log, Pow, pprint, simplify
import te_common as te

class NormSym (te.Norm):
    """Specialized norm class for symbolic calculations."""

    def __init__ (self, norm):
        super ().__init__ (norm.wtype, norm.wamount, norm.skew, norm.order)

    def __get_interval_weight_sym (self, primes):
        """Returns the weight matrix for a list of formal primes. """
        wamount = Rational (self.wamount).limit_denominator (1e3)
        match self.wtype:
            case "tenney":
                warnings.warn ("transcendental weight can be slow. Main optimizer recommended. ")
                weight_vec = Matrix (primes).applyfunc (lambda q: log (q, 2))
            case "wilson" | "benedetti":
                weight_vec = Matrix (primes)
            case "equilateral":
                weight_vec = Matrix.ones (len (primes), 1)
            # case "hahn24": #pending better implementation
            #     weight_vec = Matrix (primes).applyfunc (lambda q: 1/floor (log (24, q)))
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                self.wtype = "tenney"
                return self.__get_interval_weight_sym (primes)
        return Matrix.diag (*weight_vec.applyfunc (lambda wi: Pow (wi, wamount)))

    def __get_tuning_weight_sym (self, primes):
        return self.__get_interval_weight_sym (primes).inv ()

    def __get_interval_skew_sym (self, primes):
        skew = Rational (self.skew).limit_denominator (1e3)
        if self.skew == 0:
            return Matrix.eye (len (primes))
        else:
            return Matrix.eye (len (primes)).col_join (
                self.skew*Matrix.ones (1, len (primes)))

    def __get_tuning_skew_sym (self, primes):
        if self.skew == 0:
            return Matrix.eye (len (primes))
        elif self.skew == np.inf:
            raise NotImplementedError ("Infinite skew not supported yet.")
        else:
            skew = Rational (self.skew).limit_denominator (1e3)
        r = 1/(len (primes)*skew + 1/skew)
        kr = 1/(len (primes) + 1/skew**2)
        return (Matrix.eye (len (primes)) 
            - kr*Matrix.ones (len (primes), len (primes))).row_join (
                r*Matrix.ones (len (primes), 1))

    def tuning_x_sym (self, main, subgroup):
        primes = Matrix ([Rational (r.num, r.den) for r in subgroup.ratios ()])
        return main @ self.__get_tuning_weight_sym (primes) @ self.__get_tuning_skew_sym (primes)

    def interval_x_sym (self, main, subgroup):
        primes = Matrix ([Rational (r.num, r.den) for r in subgroup.ratios ()])
        return self.__get_interval_skew_sym (primes) @ self.__get_interval_weight_sym (primes) @ main
    
    def weightskew (self, subgroup):
        primes = Matrix ([Rational (r.num, r.den) for r in subgroup.ratios ()])
        return self.__get_tuning_weight_sym (primes) @ self.__get_tuning_skew_sym (primes)

def wrapper_symbolic (breeds, subgroup = None, norm = te.Norm (), inharmonic = False, 
        constraint = None, destretch = None, show = True):
    """
    Returns the generator tuning map, tuning map, and error map. 
    Inharmonic/subgroup modes can be configured here, 
    and the result can be displayed. 
    """
    # NOTE: "map" is a reserved word
    # optimization is preferably done in the unit of octaves, but for precision reasons

    def __mean (main):
        """
        This mean rejects the extra dimension from the denominator
        such that when skew = 0, introducing the extra dimension doesn't change the result.
        """
        return np.sum (main)/(main.size - (1 if norm.skew else 0))

    def __power_mean_norm (main):
        if norm.order == np.inf:
            return np.max (main)
        else:
            return __mean (np.fabs (main)**norm.order)**(1/norm.order)

    breeds, subgroup = te.setup (breeds, subgroup, axis = te.AXIS.ROW)
    if (inharmonic or subgroup.is_prime ()
            or norm.wtype == "tenney" and subgroup.is_prime_power ()):
        gen, tuning_projection, tempered_tuning_map, error_projection, error_map = optimizer_symbolic (
            breeds, target = subgroup, norm = norm, 
            constraint = constraint, destretch = destretch, show = show)
        error_map_x = norm.tuning_x (error_map, subgroup)
        # print (error_map_x) #for debugging
        error = __power_mean_norm (error_map_x)
        bias = __mean (error_map_x)
    else:
        subgroup_mp = subgroup.minimal_prime_subgroup ()
        subgroup2mp = subgroup.basis_matrix_to (subgroup_mp)
        breeds_mp = te.antinullspace (subgroup2mp @ te.nullspace (breeds))

        gen_mp, tuning_projection_mp, tempered_tuning_map_mp, error_projection_mp, error_map_mp = optimizer_symbolic (
            breeds_mp, target = subgroup_mp, norm = norm, 
            constraint = constraint, destretch = destretch, show = show)
        error_map_mp_x = norm.tuning_x (error_map_mp, subgroup_mp)
        # print (error_map_mp_x) #for debugging
        error = __power_mean_norm (error_map_mp_x)
        bias = __mean (error_map_mp_x)

        tempered_tuning_map = tempered_tuning_map_mp @ subgroup2mp
        gen = tempered_tuning_map @ linalg.pinv (breeds)
        error_map = tempered_tuning_map - subgroup.just_tuning_map (scalar = te.SCALAR.CENT)
        tuning_projection = Matrix (subgroup2mp).pinv () @ tuning_projection_mp @ Matrix (subgroup2mp)
        error_projection = Matrix (subgroup2mp).pinv () @ error_projection_mp @ Matrix (subgroup2mp)

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")
        if norm.wtype in te.ALGEBRAIC_WEIGHT_LIST and destretch is None:
            print ("Tuning projection map: ")
            pprint (tuning_projection)
            print ("Error projection map: ")
            pprint (error_projection)
            print ("Unchanged intervals: ")

            # this returns the eigenvalue, number of eigenvectors, 
            # and eigenvectors for each eigenvalue
            # but we're only interested in eigenvectors of unit eigenvalue
            frac_unit_eigenmonzos = tuning_projection.eigenvects ()[-1][-1]
            unit_eigenmonzos = np.column_stack ([te.matrix2array (entry) for entry in frac_unit_eigenmonzos])
            te.show_monzo_list (unit_eigenmonzos, subgroup)
        else:
            print ("Transcendental projection maps not shown. ")
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    return gen, tempered_tuning_map, error_map

def optimizer_symbolic (breeds, target = None, norm = te.Norm (), 
        constraint = None, destretch = None, show = True):
    """Returns the generator tuning map, tuning map, and error map inharmonically. """

    breeds, target = te.setup (breeds, target, axis = te.AXIS.ROW)
    norm = NormSym (norm)
    if norm.order != 2:
        raise ValueError ("Euclidean norm is required for symbolic solution. ")

    just_tuning_map = te.SCALAR.CENT*Matrix ([target.ratios (evaluate = True)]).applyfunc (lambda si: log (si, 2))
    weightskew = norm.weightskew (target)
    breeds_copy = Matrix (breeds)
    breeds_x = norm.tuning_x_sym (breeds_copy, target)

    if constraint is None:
        tuning_projection = weightskew @ breeds_x.pinv () @ breeds_x @ weightskew.pinv ()
    else:
        cons_monzo_list = Matrix (constraint.basis_matrix_to (target))
        cons_monzo_list_x = norm.interval_x_sym (cons_monzo_list, target)

        # orthonormal complement basis of the weight-skewed constraints
        comp_monzo_list_x = Matrix (BlockMatrix (Matrix.orthogonalize (
            *cons_monzo_list_x.T.nullspace (), normalize = True)))
        
        # weight-skewed working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weight-skewed constraint and its orthonormal complement
        subgroup_x = cons_monzo_list_x.row_join (comp_monzo_list_x)

        # weight-skewed map and constraints in the working basis
        breeds_xs = Matrix (breeds_x @ subgroup_x).rref ()[0]
        cons_monzo_list_xs = subgroup_x.inv () @ cons_monzo_list_x

        # get the weight-skewed tuning projection map in the working basis and copy the first r columns
        tuning_projection_xs = breeds_xs.pinv () @ breeds_xs
        tuning_projection_xs_eigen = tuning_projection_xs @ cons_monzo_list_xs

        # find the minor tuning projection map
        r = cons_monzo_list.rank ()
        breeds_xs_minor = breeds_xs[r:, r:]
        tuning_projection_xs_minor = breeds_xs_minor.pinv () @ breeds_xs_minor

        # compose the inverse of weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs_inv = tuning_projection_xs_eigen.row_join (
            Matrix.zeros (r, breeds_xs_minor.shape[1]).col_join (tuning_projection_xs_minor))
        
        # weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs = tuning_projection_xs_inv.pinv ()

        # remove weight-skew and basis transformation
        tuning_projection = simplify (
            weightskew @ subgroup_x @ tuning_projection_xs @ subgroup_x.inv () @ weightskew.pinv ())
    if show: 
        print ("Solved. ")

    if destretch is not None:
        des_monzo = Matrix (te.ratio2monzo (te.as_ratio (destretch), subgroup = target))
        if (tempered_size := (just_tuning_map @ tuning_projection @ des_monzo).det ()) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            tuning_projection *= (just_tuning_map @ des_monzo).det ()/tempered_size

    gen = np.array (just_tuning_map @ tuning_projection @ breeds_copy.pinv (), dtype = float).squeeze ()
    tempered_tuning_map = np.array (just_tuning_map @ tuning_projection, dtype = float).squeeze ()
    error_projection = tuning_projection - Matrix.eye (len (target))
    error_map = np.array (just_tuning_map @ error_projection, dtype = float).squeeze ()

    return gen, tuning_projection, tempered_tuning_map, error_projection, error_map
