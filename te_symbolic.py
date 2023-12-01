# © 2020-2023 Flora Canou | Version 1.0.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import Rational, log, Pow, pprint, simplify
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class NormSym (te.Norm):
    """Specialized norm class for symbolic calculations."""

    def __init__ (self, norm):
        super ().__init__ (norm.wtype, norm.wamount, norm.skew, norm.order)

    def __get_weight_sym (self, primes):
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
            #     weight_vec = Matrix (subgroup).applyfunc (lambda q: ceil (log (q, 24)))
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                self.wtype = "tenney"
                return self.__get_weight_sym (primes)
        return Matrix.diag (*weight_vec.applyfunc (lambda wi: Pow (wi, wamount)))

    def __get_dual_weight_sym (self, primes):
        return self.__get_weight_sym (primes).inv ()

    def __get_skew_sym (self, primes):
        skew = Rational (self.skew).limit_denominator (1e3)
        if self.skew == 0:
            return Matrix.eye (len (primes))
        else:
            return Matrix.eye (len (primes)).col_join (
                Matrix.ones (len (primes), 1)
            )

    def __get_dual_skew_sym (self, primes):
        skew = Rational (self.skew).limit_denominator (1e3)
        if self.skew == 0:
            return Matrix.eye (len (primes))
        else:
            return (Matrix.eye (len (primes)) 
                - (skew**2/(len (primes)*skew**2 + 1))*Matrix.ones (len (primes), len (primes))).row_join (
                (skew/(len (primes)*skew**2 + 1))*Matrix.ones (len (primes), 1)
            )

    def tuning_x_sym (self, main, subgroup):
        primes = Matrix (subgroup.ratios (evaluate = True))
        return main @ self.__get_dual_weight_sym (primes) @ self.__get_dual_skew_sym (primes)

    def interval_x_sym (self, main, subgroup):
        primes = Matrix (subgroup.ratios (evaluate = True))
        return self.__get_skew_sym (primes) @ self.__get_weight_sym (primes) @ main
    
    def weightskew (self, subgroup):
        primes = Matrix (subgroup.ratios (evaluate = True))
        return self.__get_dual_weight_sym (primes) @ self.__get_dual_skew_sym (primes)

def wrapper_symbolic (vals, subgroup = None, norm = te.Norm (), inharmonic = False, 
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
            return np.power (__mean (np.power (np.abs (main), norm.order)), np.reciprocal (float (norm.order)))

    vals, subgroup = te.setup (vals, subgroup, axis = te.AXIS.ROW)
    if subgroup.is_simple () or inharmonic:
        gen, tuning_projection, tempered_tuning_map, error_projection, error_map = symbolic (
            vals, target = subgroup, norm = norm, 
            constraint = constraint, destretch = destretch
        )
        error_map_x = norm.tuning_x (error_map, subgroup)
        # print (error_map_x) #for debugging
        error = __power_mean_norm (error_map_x)
        bias = __mean (error_map_x)
    else:
        vals_parent = te.antinullspace (subgroup.basis_matrix @ te.nullspace (vals))
        subgroup_parent = te.get_subgroup (subgroup.basis_matrix, axis = te.AXIS.COL)

        gen_parent, tuning_projection_parent, tempered_tuning_map_parent, error_projection_parent, error_map_parent = symbolic (
            vals_parent, target = subgroup_parent, norm = norm, 
            constraint = constraint, destretch = destretch
        )

        error_map_parent_x = norm.tuning_x (error_map_parent, subgroup_parent)
        # print (error_map_parent_x) #for debugging
        error = __power_mean_norm (error_map_parent_x)
        bias = __mean (error_map_parent_x)

        tempered_tuning_map = tempered_tuning_map_parent @ subgroup.basis_matrix
        gen = tempered_tuning_map @ linalg.pinv (vals)
        error_map = tempered_tuning_map - subgroup.just_tuning_map (scalar = te.SCALAR.CENT)
        tuning_projection = Matrix (subgroup.basis_matrix).pinv () @ tuning_projection_parent @ Matrix (subgroup.basis_matrix)
        error_projection = Matrix (subgroup.basis_matrix).pinv () @ error_projection_parent @ Matrix (subgroup.basis_matrix)

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")
        if norm.wtype in te.ALGEBRAIC_WEIGHT_LIST and destretch is None:
            print ("Tuning projection map: ")
            pprint (tuning_projection)
            print ("Error projection map: ")
            pprint (error_projection)
            print ("Eigenmonzos: ")
            eigenmonzo_list = np.column_stack (tuning_projection.eigenvects ()[-1][-1])
            te.show_monzo_list (eigenmonzo_list, subgroup)
        else:
            print ("Transcendental projection maps not shown. ")

    return gen, tempered_tuning_map, error_map

def symbolic (vals, target = None, norm = te.Norm (), 
        constraint = None, destretch = None, show = True, *, 
        subgroup = None, cons_monzo_list = None, des_monzo = None): #deprecated parameters
    # NOTE: "map" is a reserved word
    # optimization is preferably done in the unit of octaves, but for precision reasons
    
    if not subgroup is None:
        warnings.warn ("\"subgroup\" is deprecated. Use \"target\" instead. ")
        target = te.Subgroup (subgroup)
    if not cons_monzo_list is None:
        warnings.warn ("\"cons_monzo_list\" is deprecated. Use \"constraint\" instead. ")
        constraint = te.Subgroup ([te.monzo2ratio (entry) for entry in cons_monzo_list.T])
    if not des_monzo is None:
        warnings.warn ("\"des_monzo\" is deprecated. Use \"destretch\" instead. ")
        destretch = te.monzo2ratio (des_monzo)

    vals, target = te.setup (vals, target, axis = te.AXIS.ROW)
    norm = NormSym (norm)
    if norm.order != 2:
        raise ValueError ("Euclidean norm is required for symbolic solution. ")

    just_tuning_map = te.SCALAR.CENT*Matrix ([target.ratios (evaluate = True)]).applyfunc (lambda si: log (si, 2))
    weightskew = norm.weightskew (target)
    vals_copy = Matrix (vals)
    vals_x = norm.tuning_x_sym (vals_copy, target)

    if constraint is None:
        tuning_projection = weightskew @ vals_x.pinv () @ vals_x @ weightskew.pinv ()
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
        vals_xs = Matrix (vals_x @ subgroup_x).rref ()[0]
        cons_monzo_list_xs = subgroup_x.inv () @ cons_monzo_list_x
        # gets the weight-skewed tuning projection map in the working basis and copies the first r columns
        tuning_projection_xs = vals_xs.pinv () @ vals_xs
        tuning_projection_xs_eigen = tuning_projection_xs @ cons_monzo_list_xs

        # finds the minor tuning projection map
        r = cons_monzo_list.rank ()
        vals_xs_minor = vals_xs[r:, r:]
        tuning_projection_xs_minor = vals_xs_minor.pinv () @ vals_xs_minor
        # composes the inverse of weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs_inv = tuning_projection_xs_eigen.row_join (
            Matrix.zeros (r, vals_xs_minor.shape[1]).col_join (tuning_projection_xs_minor))
        # weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs = tuning_projection_xs_inv.pinv ()
        # removes weight-skew and basis transformation
        tuning_projection = simplify (
            weightskew @ subgroup_x @ tuning_projection_xs @ subgroup_x.inv () @ weightskew.pinv ())
    print ("Solved. ")

    if not destretch is None:
        des_monzo = Matrix (te.ratio2monzo (te.as_ratio (destretch), subgroup = target))
        if (tempered_size := (just_tuning_map @ tuning_projection @ des_monzo).det ()) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            tuning_projection *= (just_tuning_map @ des_monzo).det ()/tempered_size

    gen = np.array (just_tuning_map @ tuning_projection @ vals_copy.pinv (), dtype = float).squeeze ()
    tempered_tuning_map = np.array (just_tuning_map @ tuning_projection, dtype = float).squeeze ()
    error_projection = tuning_projection - Matrix.eye (len (target))
    error_map = np.array (just_tuning_map @ error_projection, dtype = float).squeeze ()

    return gen, tuning_projection, tempered_tuning_map, error_projection, error_map
