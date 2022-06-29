# © 2020-2022 Flora Canou | Version 0.20.0
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from sympy.matrices import Matrix, BlockMatrix
from sympy import lcm, pprint
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

# specialized weighter for symbolic calculations
def get_sym_weight (subgroup, wtype = "frobenius"):
    if wtype == "frobenius":
        return Matrix.eye (len (subgroup))
    elif wtype == "benedetti":
        return Matrix (len (subgroup), len (subgroup), lambda i, j: lcm (subgroup)/subgroup[i] if i == j else 0)
    else:
        warnings.warn ("weighter type not supported, using default (\"frobenius\")")
        return get_sym_weight (subgroup, wtype = "frobenius")

def symbolic (map, subgroup = None, wtype = "frobenius",
        cons_monzo_list = None, show = True):
    map, subgroup = te.get_subgroup (np.array (map), subgroup, axis = te.ROW)
    weight = get_sym_weight (subgroup, wtype)
    map_copy = Matrix (map)
    map_w = map_copy @ weight

    if cons_monzo_list is None:
        projection = weight @ map_w.pinv () @ map_w @ weight.inv ()
    else:
        cons_monzo_list_copy = Matrix (cons_monzo_list)
        cons_monzo_list_w = weight.inv () @ cons_monzo_list_copy
        # orthonormal complement basis of the weighted constraints
        comp_monzo_list_w = Matrix (BlockMatrix (Matrix.orthogonalize (*cons_monzo_list_w.T.nullspace (), normalize = True)))
        # weighted working subgroup basis in terms of monzo list, isomorphic to the original
        # joined by weighted constraint and its orthonormal complement
        subgroup_wt = cons_monzo_list_w.row_join (comp_monzo_list_w)

        # weighted map in the working basis
        map_wt = Matrix (map_w @ subgroup_wt).rref ()[0]
        # projection map in the working basis
        projection_wt = map_wt.pinv () @ map_wt

        # weighted constraints in the working basis
        cons_monzo_list_wt = subgroup_wt.inv () @ cons_monzo_list_w
        # copies the first r columns of the projection map
        projection_wt_eigen = projection_wt @ cons_monzo_list_wt

        r = cons_monzo_list_copy.rank ()
        map_wt_minor = map_wt[r:, r:]
        projection_wt_minor = map_wt_minor.pinv () @ map_wt_minor
        projection_wt_inv = projection_wt_eigen.row_join (Matrix.zeros (r, len (subgroup) - r).col_join (projection_wt_minor))
        # weighted projection map for constrained tuning in the working basis
        projection_wt = projection_wt_inv.pinv ()
        # removes weight and basis transformation
        projection = weight @ subgroup_wt @ projection_wt @ subgroup_wt.inv () @ weight.inv ()

    print ("Solved. ")
    jip = np.log2 (subgroup)*te.SCALAR
    gen = jip @ np.array (projection @ map_copy.pinv (), dtype = float)
    tuning_map = jip @ np.array (projection, dtype = float)
    misprojection = projection - Matrix.eye (len (subgroup))
    mistuning_map = jip @ np.array (misprojection, dtype = float)

    if show:
        print (f"Generators: {gen} (¢)", "Projection map: ", sep = "\n")
        pprint (projection)
        print (f"Tuning map: {tuning_map} (¢)", "Misprojection map: ", sep = "\n")
        pprint (misprojection)
        print (f"Mistuning map: {mistuning_map} (¢)")

    return gen, tuning_map, mistuning_map
