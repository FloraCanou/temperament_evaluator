# © 2020-2026 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import Rational, log, Pow, Mod, pprint, simplify
from . import te_common as te

class NormSym (te.Norm):
    """Specialized norm class for symbolic calculations."""

    def __init__ (self, norm):
        super ().__init__ (wmode = norm.wmode, wstrength = norm.wstrength, 
            skew = norm.skew, order = norm.order)

    def __weight_vec_sym (self, primes):
        """Returns the interval weight vector for a list of formal primes. """

        if not isinstance (self.wmode, (int, np.integer)):
            raise TypeError ("non-integer modes not supported. ")
        wmode = self.wmode if self.wstrength else 0
        wstrength = Rational (self.wstrength).limit_denominator (1e3)

        # test shows that besides mode 0, only mode +/-1
        # with integer or half-integer strengths demonstrate acceptable performance
        if wmode not in (-1, 0, 1): 
            warnings.warn ("remote weight modes can be slow. " \
                "Main optimizer recommended. ")
        if Mod (wstrength, Rational (1, 2)): 
            warnings.warn ("non-integer, non-half-integer weight strengths can be slow. " \
                "Main optimizer recommended. ")

        def modal_weighter (primes, m): 
            if m == 0: 
                return primes
            elif m > 0: 
                return modal_weighter (primes.applyfunc (lambda q: 2*log (q, 2)), m - 1)
            else: 
                return modal_weighter (primes.applyfunc (lambda q: 2**(q/2)), m + 1)

        return modal_weighter (Matrix (primes), wmode).applyfunc (lambda wi: Pow (wi/2, wstrength))

    def interval_weight_sym (self, primes):
        """Returns the interval weight matrix for a list of formal primes. """
        return Matrix.diag (*self.__weight_vec_sym (primes))

    def val_weight_sym (self, primes):
        """Returns the val weight matrix for a list of formal primes. """
        return Matrix.diag (*self.__weight_vec_sym (primes).applyfunc (lambda wi: 1/wi))

    def interval_skew_sym (self, primes):
        """Returns the interval skew matrix for a list of formal primes. """
        skew = Rational (self.skew).limit_denominator (1e3)
        if self.skew == 0:
            return Matrix.eye (len (primes))
        else:
            return Matrix.eye (len (primes)).col_join (
                self.skew*Matrix.ones (1, len (primes)))

    def val_skew_sym (self, primes):
        """Returns the val skew matrix for a list of formal primes. """
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

    def val_transform_sym (self, vals, subgroup):
        primes = Matrix ([Rational (r.num, r.den) for r in subgroup.ratios ()])
        return vals @ self.val_weight_sym (primes) @ self.val_skew_sym (primes)

    def interval_transform_sym (self, intervals, subgroup):
        primes = Matrix ([Rational (r.num, r.den) for r in subgroup.ratios ()])
        return self.interval_skew_sym (primes) @ self.interval_weight_sym (primes) @ intervals

    def val_transformer (self, subgroup):
        primes = Matrix ([Rational (r.num, r.den) for r in subgroup.ratios ()])
        return self.val_weight_sym (primes) @ self.val_skew_sym (primes)

def wrapper_sym (breeds, target = None, norm = te.Norm (), inharmonic = False, 
        constraint = None, destretch = None, show = True, *, subgroup = None): 
    """
    Returns and displays the generator tuning map, tempered tuning map, 
    and error map in cents. Also displays the corresponding projection maps.
    """
    # NOTE: "map" is a reserved word
    # in cents for consistency with wrapper_main

    if subgroup is not None: 
        warnings.warn ("'subgroup' is deprecated. Use 'target' instead. ", FutureWarning)
        if target is None: 
            target = subgroup

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

    breeds, target = te.setup (breeds, target, axis = te.AXIS.ROW)
    if (inharmonic or target.is_prime ()
            or norm.wmode == 1 and norm.wstrength == 1 and target.is_prime_power ()):
        gen, tuning_projection, tempered_tuning_map, error_projection, error_map = __optimizer_sym (
            breeds, target, norm, constraint, destretch, show)
        error_map_x = norm.val_transform (error_map, target)
        error = __power_mean_norm (error_map_x)
        bias = __mean (error_map_x)
    else:
        breeds_mp, target_mp = te.breeds2mp (breeds, target)
        gen_mp, tuning_projection_mp, tempered_tuning_map_mp, error_projection_mp, error_map_mp = __optimizer_sym (
            breeds_mp, target_mp, norm, constraint, destretch, show)
        error_map_mp_x = norm.val_transform (error_map_mp, target_mp)
        error = __power_mean_norm (error_map_mp_x)
        bias = __mean (error_map_mp_x)

        just_tuning_map = target.just_tuning_map (scalar = te.SCALAR.CENT)
        tempered_tuning_map = tempered_tuning_map_mp @ target2mp
        gen = tempered_tuning_map @ linalg.pinv (breeds)
        error_map = tempered_tuning_map - just_tuning_map
        tuning_projection = Matrix (target2mp).pinv () @ tuning_projection_mp @ Matrix (target2mp)
        error_projection = Matrix (target2mp).pinv () @ error_projection_mp @ Matrix (target2mp)

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")
        if (norm.wmode == 0 or norm.wstrength == 0) and destretch is None:
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
            te.show_monzo_list (unit_eigenmonzos, target)
        else:
            print ("Projection maps not shown. ")
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    return gen, tempered_tuning_map, error_map

def __optimizer_sym (breeds, target, norm, constraint, destretch, show): 
    """
    Returns the optimal generator tuning map, tempered tuning map, 
    and error map inharmonically in cents. 
    """

    norm = NormSym (norm)
    if norm.order != 2:
        raise ValueError ("Euclidean norm is required for symbolic solution. ")

    just_tuning_map = te.SCALAR.CENT*Matrix ([target.ratios (evaluate = True)]).applyfunc (lambda si: log (si, 2))
    val_transformer = norm.val_transformer (target)
    breeds_copy = Matrix (breeds)
    breeds_x = norm.val_transform_sym (breeds_copy, target)

    if constraint is None:
        tuning_projection = val_transformer @ breeds_x.pinv () @ breeds_x @ val_transformer.pinv ()
    else:
        cons_basis_matrix = Matrix (constraint.basis_matrix_to (target))
        cons_basis_matrix_x = norm.interval_transform_sym (cons_basis_matrix, target)

        # orthonormal complement basis of the weight-skewed constraints
        comp_monzo_list_x = Matrix (BlockMatrix (Matrix.orthogonalize (
            *cons_basis_matrix_x.T.nullspace (), normalize = True)))
        
        # weight-skewed working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weight-skewed constraint and its orthonormal complement
        target_x = cons_basis_matrix_x.row_join (comp_monzo_list_x)

        # weight-skewed map and constraints in the working basis
        breeds_xs = Matrix (breeds_x @ target_x).rref ()[0]
        cons_basis_matrix_xs = target_x.inv () @ cons_basis_matrix_x

        # get the weight-skewed tuning projection map in the working basis and copy the first r columns
        tuning_projection_xs = breeds_xs.pinv () @ breeds_xs
        tuning_projection_xs_eigen = tuning_projection_xs @ cons_basis_matrix_xs

        # find the minor tuning projection map
        r = cons_basis_matrix.rank ()
        breeds_xs_minor = breeds_xs[r:, r:]
        tuning_projection_xs_minor = breeds_xs_minor.pinv () @ breeds_xs_minor

        # compose the inverse of weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs_inv = tuning_projection_xs_eigen.row_join (
            Matrix.zeros (r, breeds_xs_minor.shape[1]).col_join (tuning_projection_xs_minor))
        
        # weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs = tuning_projection_xs_inv.pinv ()

        # remove weight-skew and basis transformation
        tuning_projection = simplify (
            val_transformer @ target_x @ tuning_projection_xs @ target_x.inv () @ val_transformer.pinv ())
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
