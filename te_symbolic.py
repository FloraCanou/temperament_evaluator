# © 2020-2022 Flora Canou | Version 0.21.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from sympy.matrices import Matrix, BlockMatrix
from sympy import Rational, log, pprint, simplify
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

# specialized weight-skew matrices for symbolic calculations
def get_weight_sym (subgroup, wtype = "tenney"):
    if wtype == "tenney":
        warnings.warn ("transcendental weight can be slow. Main optimizer recommended. ")
        return Matrix (len (subgroup), len (subgroup),
            lambda i, j: 1/log (subgroup[i], 2) if i == j else 0)
    elif wtype == "frobenius":
        return Matrix.eye (len (subgroup))
    elif wtype == "benedetti":
        return Matrix (len (subgroup), len (subgroup),
            lambda i, j: 1/Rational (subgroup[i]) if i == j else 0)
    else:
        warnings.warn ("weighter type not supported, using default (\"tenney\")")
        return get_weight_sym (subgroup, wtype = "tenney")

def get_skew_sym (subgroup, skew):
    skew = Rational (skew).limit_denominator (1e3)
    return (Matrix.eye (len (subgroup)) - (skew**2/(len (subgroup)*skew**2 + 1))*Matrix.ones (len (subgroup), len (subgroup))).row_join (
        (skew/(len (subgroup)*skew**2 + 1))*Matrix.ones (len (subgroup), 1))

def symbolic (map, subgroup = None, wtype = "frobenius", skew = 0,
        cons_monzo_list = None, des_monzo = None, show = True):
    map, subgroup = te.get_subgroup (np.array (map), subgroup, axis = te.ROW)

    jip = Matrix ([subgroup]).applyfunc (lambda si: log (si, 2))*te.SCALAR
    weightskew = get_weight_sym (subgroup, wtype) @ get_skew_sym (subgroup, skew)
    map_copy = Matrix (map)
    map_wx = map_copy @ weightskew

    if cons_monzo_list is None:
        projection = weightskew @ map_wx.pinv () @ map_wx @ weightskew.pinv ()
    else:
        cons_monzo_list_copy = Matrix (cons_monzo_list)
        cons_monzo_list_wx = weightskew.pinv () @ cons_monzo_list_copy
        # orthonormal complement basis of the weight-skewed constraints
        comp_monzo_list_wx = Matrix (BlockMatrix (Matrix.orthogonalize (*cons_monzo_list_wx.T.nullspace (), normalize = True)))
        # weight-skewed working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weight-skewed constraint and its orthonormal complement
        subgroup_wx = cons_monzo_list_wx.row_join (comp_monzo_list_wx)

        # weight-skewed map and constraints in the working basis
        map_wxs = Matrix (map_wx @ subgroup_wx).rref ()[0]
        cons_monzo_list_wxs = subgroup_wx.inv () @ cons_monzo_list_wx
        # gets the weight-skewed projection map in the working basis and copies the first r columns
        projection_wxs = map_wxs.pinv () @ map_wxs
        projection_wxs_eigen = projection_wxs @ cons_monzo_list_wxs

        # finds the minor projection map
        r = cons_monzo_list_copy.rank ()
        map_wxs_minor = map_wxs[r:, r:]
        projection_wxs_minor = map_wxs_minor.pinv () @ map_wxs_minor
        # composes the inverse of weight-skewed constrained projection map in the working basis
        projection_wxs_inv = projection_wxs_eigen.row_join (Matrix.zeros (r, map_wxs_minor.shape[1]).col_join (projection_wxs_minor))
        # weight-skewed constrained projection map in the working basis
        projection_wxs = projection_wxs_inv.pinv ()
        # removes weight-skew and basis transformation
        projection = simplify (weightskew @ subgroup_wx @ projection_wxs @ subgroup_wx.inv () @ weightskew.pinv ())
    print ("Solved. ")

    if not des_monzo is None:
        des_monzo_copy = Matrix (des_monzo)
        if des_monzo_copy.rank () > 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (tempered_size := (jip @ projection @ des_monzo_copy).det ()) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            projection *= (jip @ des_monzo_copy).det ()/tempered_size

    gen = np.array (jip @ projection @ map_copy.pinv (), dtype = float).squeeze ()
    tuning_map = np.array (jip @ projection, dtype = float).squeeze ()
    misprojection = projection - Matrix.eye (len (subgroup))
    mistuning_map = np.array (jip @ misprojection, dtype = float).squeeze ()
    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")
        if wtype in te.ALGEBRAIC_WEIGHT_LIST and des_monzo is None:
            print ("Projection map: ")
            pprint (projection)
            print ("Misprojection map: ")
            pprint (misprojection)
        else:
            print ("Transcendental projection map and misprojection map not shown. ")

    return gen, tuning_map, mistuning_map
