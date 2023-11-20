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

    def get_weight_sym (self, subgroup):
        wamount = Rational (self.wamount).limit_denominator (1e3)
        if self.wtype == "tenney":
            warnings.warn ("transcendental weight can be slow. Main optimizer recommended. ")
            weight_vec = Matrix (subgroup.ratios (evaluate = True)).applyfunc (lambda si: log (2, si))
        elif self.wtype == "wilson" or self.wtype == "benedetti":
            weight_vec = Matrix (subgroup.ratios (evaluate = True)).applyfunc (lambda si: 1/si)
        elif self.wtype == "equilateral":
            weight_vec = Matrix.ones (len (subgroup), 1)
        # elif self.wtype == "hahn24": #pending better implementation
        #     weight_vec = Matrix (subgroup).applyfunc (lambda si: floor (log (24, si)))
        else:
            warnings.warn ("weighter type not supported, using default (\"tenney\")")
            self.wtype = "tenney"
            return self.get_weight_sym (subgroup)
        return Matrix.diag (*weight_vec.applyfunc (lambda wi: Pow (wi, wamount)))

    def get_skew_sym (self, subgroup):
        skew = Rational (self.skew).limit_denominator (1e3)
        return (Matrix.eye (len (subgroup)) 
            - (skew**2/(len (subgroup)*skew**2 + 1))*Matrix.ones (len (subgroup), len (subgroup))).row_join (
            (skew/(len (subgroup)*skew**2 + 1))*Matrix.ones (len (subgroup), 1))

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

    vals, target = te.get_subgroup (vals, target, axis = te.AXIS.ROW)
    norm = NormSym (norm)
    if norm.order != 2:
        raise ValueError ("Euclidean norm is required for symbolic solution. ")

    just_tuning_map = te.SCALAR.CENT*Matrix ([target.ratios (evaluate = True)]).applyfunc (lambda si: log (si, 2))
    weightskew = norm.get_weight_sym (target) @ norm.get_skew_sym (target)
    vals_copy = Matrix (vals)
    vals_x = vals_copy @ weightskew

    if constraint is None:
        projection = weightskew @ vals_x.pinv () @ vals_x @ weightskew.pinv ()
    else:
        cons_monzo_list = Matrix (constraint.basis_matrix_to (target))
        cons_monzo_list_x = weightskew.pinv () @ cons_monzo_list
        # orthonormal complement basis of the weight-skewed constraints
        comp_monzo_list_x = Matrix (BlockMatrix (Matrix.orthogonalize (
            *cons_monzo_list_x.T.nullspace (), normalize = True)))
        # weight-skewed working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weight-skewed constraint and its orthonormal complement
        subgroup_x = cons_monzo_list_x.row_join (comp_monzo_list_x)

        # weight-skewed map and constraints in the working basis
        vals_xs = Matrix (vals_x @ subgroup_x).rref ()[0]
        cons_monzo_list_xs = subgroup_x.inv () @ cons_monzo_list_x
        # gets the weight-skewed projection map in the working basis and copies the first r columns
        projection_xs = vals_xs.pinv () @ vals_xs
        projection_xs_eigen = projection_xs @ cons_monzo_list_xs

        # finds the minor projection map
        r = cons_monzo_list.rank ()
        vals_xs_minor = vals_xs[r:, r:]
        projection_xs_minor = vals_xs_minor.pinv () @ vals_xs_minor
        # composes the inverse of weight-skewed constrained projection map in the working basis
        projection_xs_inv = projection_xs_eigen.row_join (
            Matrix.zeros (r, vals_xs_minor.shape[1]).col_join (projection_xs_minor))
        # weight-skewed constrained projection map in the working basis
        projection_xs = projection_xs_inv.pinv ()
        # removes weight-skew and basis transformation
        projection = simplify (
            weightskew @ subgroup_x @ projection_xs @ subgroup_x.inv () @ weightskew.pinv ())
    print ("Solved. ")

    if not destretch is None:
        des_monzo = Matrix (te.ratio2monzo (te.as_ratio (destretch), subgroup = target))
        if (tempered_size := (just_tuning_map @ projection @ des_monzo).det ()) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            projection *= (just_tuning_map @ des_monzo).det ()/tempered_size

    gen = np.array (just_tuning_map @ projection @ vals_copy.pinv (), dtype = float).squeeze ()
    tempered_tuning_map = np.array (just_tuning_map @ projection, dtype = float).squeeze ()
    misprojection = projection - Matrix.eye (len (target))
    mistuning_map = np.array (just_tuning_map @ misprojection, dtype = float).squeeze ()

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")
        if norm.wtype in te.ALGEBRAIC_WEIGHT_LIST and des_monzo is None:
            print ("Projection map: ")
            pprint (projection)
            print ("Misprojection map: ")
            pprint (misprojection)
            print ("Eigenmonzos: ")
            eigenmonzo_list = projection.eigenvects ()[-1][-1]
            te.show_monzo_list (eigenmonzo_list, target)
        else:
            print ("Transcendental projection map and misprojection map not shown. ")

    return gen, tempered_tuning_map, mistuning_map
