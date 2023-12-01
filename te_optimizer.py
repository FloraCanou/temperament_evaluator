# © 2020-2023 Flora Canou | Version 1.0.0
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
from sympy.matrices import Matrix, normalforms
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def wrapper_main (vals, subgroup = None, norm = te.Norm (), inharmonic = False, 
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
        gen = optimizer_main (
            vals, target = subgroup, norm = norm, 
            constraint = constraint, destretch = destretch
        )
        tempered_tuning_map = gen @ vals
        error_map = tempered_tuning_map - subgroup.just_tuning_map (scalar = te.SCALAR.CENT)

        error_map_x = norm.tuning_x (error_map, subgroup)
        # print (error_map_x) #for debugging
        error = __power_mean_norm (error_map_x)
        bias = __mean (error_map_x)
    else:
        vals_parent = te.antinullspace (subgroup.basis_matrix @ te.nullspace (vals))
        subgroup_parent = te.get_subgroup (subgroup.basis_matrix, axis = te.AXIS.COL)

        gen_parent = optimizer_main (
            vals_parent, target = subgroup_parent, norm = norm, 
            constraint = constraint, destretch = destretch
        )
        tempered_tuning_map_parent = gen_parent @ vals_parent
        error_map_parent = tempered_tuning_map_parent - subgroup_parent.just_tuning_map (scalar = te.SCALAR.CENT)

        error_map_parent_x = norm.tuning_x (error_map_parent, subgroup_parent)
        # print (error_map_parent_x) #for debugging
        error = __power_mean_norm (error_map_parent_x)
        bias = __mean (error_map_parent_x)

        tempered_tuning_map = tempered_tuning_map_parent @ subgroup.basis_matrix
        gen = tempered_tuning_map @ linalg.pinv (vals)
        error_map = tempered_tuning_map - subgroup.just_tuning_map (scalar = te.SCALAR.CENT)

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    return gen, tempered_tuning_map, error_map

def optimizer_main (vals, target = None, norm = te.Norm (), 
        constraint = None, destretch = None, *, 
        subgroup = None, cons_monzo_list = None, des_monzo = None, show = True): #deprecated parameters

    if not subgroup is None:
        warnings.warn ("\"subgroup\" is deprecated. Use \"target\" instead. ")
        target = te.Subgroup (subgroup)
    if not cons_monzo_list is None:
        warnings.warn ("\"cons_monzo_list\" is deprecated. Use \"constraint\" instead. ")
        constraint = te.Subgroup ([te.monzo2ratio (entry) for entry in cons_monzo_list.T])
    if not des_monzo is None:
        warnings.warn ("\"des_monzo\" is deprecated. Use \"destretch\" instead. ")
        destretch = te.monzo2ratio (des_monzo)

    just_tuning_map = target.just_tuning_map (scalar = te.SCALAR.CENT)
    vals_x = norm.tuning_x (vals, target)
    just_tuning_map_x = norm.tuning_x (just_tuning_map, target)
    if norm.order == 2 and constraint is None: #simply using lstsq for better performance
        res = linalg.lstsq (vals_x.T, just_tuning_map_x)
        gen = res[0]
        print ("Euclidean tuning without constraints, solved using lstsq. ")
    else:
        gen0 = [te.SCALAR.CENT]*vals.shape[0] #initial guess
        if constraint is None:
            cons = ()
        else:
            cons_monzo_list = constraint.basis_matrix_to (target)
            cons = {
                'type': 'eq', 
                'fun': lambda gen: (gen @ vals - just_tuning_map) @ cons_monzo_list
            }
        res = optimize.minimize (lambda gen: linalg.norm (gen @ vals_x - just_tuning_map_x, ord = norm.order), gen0, 
            method = "SLSQP", options = {'ftol': 1e-9}, constraints = cons)
        print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if not destretch is None:
        des_monzo = te.ratio2monzo (te.as_ratio (destretch), subgroup = target)
        if (tempered_size := gen @ vals @ des_monzo) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            gen *= (just_tuning_map @ des_monzo)/tempered_size

    return gen

optimiser_main = optimizer_main
