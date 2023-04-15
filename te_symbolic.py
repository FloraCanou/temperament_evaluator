# © 2020-2023 Flora Canou | Version 0.25.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from sympy.matrices import Matrix, BlockMatrix
from sympy import Rational, log, floor, Pow, pprint, simplify
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

# specialized weight-skew matrices for symbolic calculations
def __get_weight_sym (subgroup, wtype = "tenney", wamount = 1):
    wamount = Rational (wamount).limit_denominator (1e3)
    if wtype == "tenney":
        warnings.warn ("transcendental weight can be slow. Main optimizer recommended. ")
        weight_vec = Matrix (subgroup).applyfunc (lambda si: 1/log (si, 2))
    elif wtype == "wilson" or wtype == "benedetti":
        weight_vec = Matrix (subgroup).applyfunc (lambda si: 1/si)
    elif wtype == "equilateral":
        weight_vec = Matrix.ones (len (subgroup), 1)
    else:
        warnings.warn ("weighter type not supported, using default (\"tenney\")")
        return get_weight_sym (subgroup, wtype = "tenney")
    return Matrix.diag (*weight_vec.applyfunc (lambda wi: Pow (wi, wamount)))

def __get_skew_sym (subgroup, skew, order):
    skew = Rational (skew).limit_denominator (1e3)
    return (Matrix.eye (len (subgroup)) 
        - (skew**2/(len (subgroup)*skew**2 + 1))*Matrix.ones (len (subgroup), len (subgroup))).row_join (
        (skew/(len (subgroup)*skew**2 + 1))*Matrix.ones (len (subgroup), 1))

def symbolic (vals, subgroup = None, norm = te.Norm (wtype = "equilateral"), #"map" is a reserved word
        cons_monzo_list = None, des_monzo = None, show = True):
    vals, subgroup = te.get_subgroup (vals, subgroup, axis = te.ROW)

    # DEPRECATION WARNING
    if any ((wtype, wamount, skew, order)): 
        warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
        if wtype: norm.wtype = wtype
        if wamount: norm.wamount = wamount
        if skew: norm.skew = skew
        if order: norm.order = order

    if not norm.order == 2:
        raise ValueError ("Euclidean norm is required for symbolic solution. ")

    jip = Matrix ([subgroup]).applyfunc (lambda si: log (si, 2))*te.SCALAR
    weightskew = (__get_weight_sym (subgroup, norm.wtype, norm.wamount) 
        @ __get_skew_sym (subgroup, norm.skew, norm.order))
    vals_copy = Matrix (vals)
    vals_wx = vals_copy @ weightskew

    if cons_monzo_list is None:
        projection = weightskew @ vals_wx.pinv () @ vals_wx @ weightskew.pinv ()
    else:
        cons_monzo_list_copy = Matrix (cons_monzo_list)
        cons_monzo_list_wx = weightskew.pinv () @ cons_monzo_list_copy
        # orthonormal complement basis of the weight-skewed constraints
        comp_monzo_list_wx = Matrix (BlockMatrix (Matrix.orthogonalize (
            *cons_monzo_list_wx.T.nullspace (), normalize = True)))
        # weight-skewed working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weight-skewed constraint and its orthonormal complement
        subgroup_wx = cons_monzo_list_wx.row_join (comp_monzo_list_wx)

        # weight-skewed map and constraints in the working basis
        vals_wxs = Matrix (vals_wx @ subgroup_wx).rref ()[0]
        cons_monzo_list_wxs = subgroup_wx.inv () @ cons_monzo_list_wx
        # gets the weight-skewed projection map in the working basis and copies the first r columns
        projection_wxs = vals_wxs.pinv () @ vals_wxs
        projection_wxs_eigen = projection_wxs @ cons_monzo_list_wxs

        # finds the minor projection map
        r = cons_monzo_list_copy.rank ()
        vals_wxs_minor = vals_wxs[r:, r:]
        projection_wxs_minor = vals_wxs_minor.pinv () @ vals_wxs_minor
        # composes the inverse of weight-skewed constrained projection map in the working basis
        projection_wxs_inv = projection_wxs_eigen.row_join (
            Matrix.zeros (r, map_wxs_minor.shape[1]).col_join (projection_wxs_minor))
        # weight-skewed constrained projection map in the working basis
        projection_wxs = projection_wxs_inv.pinv ()
        # removes weight-skew and basis transformation
        projection = simplify (
            weightskew @ subgroup_wx @ projection_wxs @ subgroup_wx.inv () @ weightskew.pinv ())
    print ("Solved. ")

    if not des_monzo is None:
        des_monzo_copy = Matrix (des_monzo)
        if des_monzo_copy.rank () > 1:
            raise IndexError ("only one destretch target is allowed. ")
        elif (tempered_size := (jip @ projection @ des_monzo_copy).det ()) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            projection *= (jip @ des_monzo_copy).det ()/tempered_size

    gen = np.array (jip @ projection @ vals_copy.pinv (), dtype = float).squeeze ()
    tuning_map = np.array (jip @ projection, dtype = float).squeeze ()
    misprojection = projection - Matrix.eye (len (subgroup))
    mistuning_map = np.array (jip @ misprojection, dtype = float).squeeze ()

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tuning_map} (¢)",
            f"Mistuning map: {mistuning_map} (¢)", sep = "\n")
        if norm.wtype in te.ALGEBRAIC_WEIGHT_LIST and des_monzo is None:
            print ("Projection map: ")
            pprint (projection)
            print ("Misprojection map: ")
            pprint (misprojection)
            print ("Eigenmonzos: ")
            eigenmonzo_list = projection.eigenvects ()[-1][-1]
            te.show_monzo_list (eigenmonzo_list, subgroup)
        else:
            print ("Transcendental projection map and misprojection map not shown. ")

    return gen, tuning_map, mistuning_map
