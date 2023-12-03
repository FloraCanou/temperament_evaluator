# © 2020-2023 Flora Canou | Version 0.27.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from sympy.matrices import Matrix, BlockMatrix
from sympy import Rational, log, floor, Pow, pprint, simplify
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

# specialized class for symbolic calculations
class NormSym (te.Norm):
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
            #     weight_vec = Matrix (subgroup).applyfunc (lambda q: ceil (log (q, 24)))
            case _:
                warnings.warn ("weighter type not supported, using default (\"tenney\")")
                self.wtype = "tenney"
                return self.__get_weight_sym (primes)
        return Matrix.diag (*weight_vec.applyfunc (lambda wi: Pow (wi, wamount)))

    def __get_tuning_weight_sym (self, primes):
        return self.__get_interval_weight_sym (primes).inv ()

    def __get_interval_skew_sym (self, primes):
        skew = Rational (self.skew).limit_denominator (1e3)
        if self.skew == 0:
            return Matrix.eye (len (primes))
        else:
            return Matrix.eye (len (primes)).col_join (
                Matrix.ones (len (primes), 1)
            )

    def __get_tuning_skew_sym (self, primes):
        skew = Rational (self.skew).limit_denominator (1e3)
        if self.skew == 0:
            return Matrix.eye (len (primes))
        else:
            return (Matrix.eye (len (primes)) 
                - (skew**2/(len (primes)*skew**2 + 1))*Matrix.ones (len (primes), len (primes))).row_join (
                (skew/(len (primes)*skew**2 + 1))*Matrix.ones (len (primes), 1)
            )

    def tuning_x_sym (self, main, subgroup):
        return main @ self.__get_tuning_weight_sym (subgroup) @ self.__get_tuning_skew_sym (subgroup)

    def interval_x_sym (self, main, subgroup):
        return self.__get_interval_skew_sym (subgroup) @ self.__get_interval_weight_sym (subgroup) @ main
    
    def weightskew (self, subgroup):
        primes = Matrix (subgroup)
        return self.__get_tuning_weight_sym (primes) @ self.__get_tuning_skew_sym (primes)

def symbolic (breeds, subgroup = None, norm = te.Norm (), #NOTE: "map" is a reserved word
        cons_monzo_list = None, des_monzo = None, show = True):
    breeds, subgroup = te.setup (breeds, subgroup, axis = te.AXIS.ROW)
    norm = NormSym (norm)
    if norm.order != 2:
        raise ValueError ("Euclidean norm is required for symbolic solution. ")

    just_tuning_map = Matrix ([subgroup]).applyfunc (lambda si: log (si, 2))*te.SCALAR.CENT
    weightskew = norm.weightskew (subgroup)
    breeds_copy = Matrix (breeds)
    breeds_x = norm.tuning_x_sym (breeds_copy, subgroup)

    if cons_monzo_list is None:
        tuning_projection = weightskew @ breeds_x.pinv () @ breeds_x @ weightskew.pinv ()
    else:
        cons_monzo_list_copy = Matrix (cons_monzo_list)
        cons_monzo_list_x = norm.interval_x_sym (cons_monzo_list_copy, subgroup)
        # orthonormal complement basis of the weight-skewed constraints
        comp_monzo_list_x = Matrix (BlockMatrix (Matrix.orthogonalize (
            *cons_monzo_list_x.T.nullspace (), normalize = True)))
        # weight-skewed working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weight-skewed constraint and its orthonormal complement
        subgroup_x = cons_monzo_list_x.row_join (comp_monzo_list_x)

        # weight-skewed map and constraints in the working basis
        breeds_xs = Matrix (breeds_x @ subgroup_x).rref ()[0]
        cons_monzo_list_xs = subgroup_x.inv () @ cons_monzo_list_x
        # gets the weight-skewed tuning projection map in the working basis and copies the first r columns
        tuning_projection_xs = breeds_xs.pinv () @ breeds_xs
        tuning_projection_xs_eigen = tuning_projection_xs @ cons_monzo_list_xs

        # finds the minor tuning projection map
        r = cons_monzo_list_copy.rank ()
        breeds_xs_minor = breeds_xs[r:, r:]
        tuning_projection_xs_minor = breeds_xs_minor.pinv () @ breeds_xs_minor
        # composes the inverse of weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs_inv = tuning_projection_xs_eigen.row_join (
            Matrix.zeros (r, breeds_xs_minor.shape[1]).col_join (tuning_projection_xs_minor))
        # weight-skewed constrained tuning projection map in the working basis
        tuning_projection_xs = tuning_projection_xs_inv.pinv ()
        # removes weight-skew and basis transformation
        tuning_projection = simplify (
            weightskew @ subgroup_x @ tuning_projection_xs @ subgroup_x.inv () @ weightskew.pinv ())
    print ("Solved. ")

    if not des_monzo is None:
        des_monzo_copy = Matrix (des_monzo)
        if des_monzo_copy.rank () > 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (tempered_size := (just_tuning_map @ tuning_projection @ des_monzo_copy).det ()) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            tuning_projection *= (just_tuning_map @ des_monzo_copy).det ()/tempered_size

    gen = np.array (just_tuning_map @ tuning_projection @ breeds_copy.pinv (), dtype = float).squeeze ()
    tempered_tuning_map = np.array (just_tuning_map @ tuning_projection, dtype = float).squeeze ()
    error_projection = tuning_projection - Matrix.eye (len (subgroup))
    mistuning_map = np.array (just_tuning_map @ error_projection, dtype = float).squeeze ()

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")
        if norm.wtype in te.ALGEBRAIC_WEIGHT_LIST and des_monzo is None:
            print ("Tuning projection map: ")
            pprint (tuning_projection)
            print ("Error projection map: ")
            pprint (error_projection)
            print ("Eigenmonzos: ")
            frac_eigenmonzos = tuning_projection.eigenvects ()[-1][-1]
            eigenmonzos = np.column_stack ([te.__matrix2array (entry) for entry in frac_eigenmonzos])
            te.show_monzo_list (eigenmonzos, subgroup)
        else:
            print ("Transcendental projection maps not shown. ")

    return gen, tempered_tuning_map, mistuning_map
