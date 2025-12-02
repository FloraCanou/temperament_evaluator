# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import warnings
import numpy as np
from scipy import optimize, linalg
from . import te_common as te

def wrapper_main (breeds, target = None, norm = te.Norm (), inharmonic = False, 
        constraint = None, destretch = None, show = True, *, subgroup = None):
    """
    Returns and displays the optimal generator tuning map, 
    tempered tuning map, and error map in cents. 
    """
    # NOTE: "map" is a reserved word
    # optimization would ideally be performed in the unit of octaves
    # unfortunately, that often results in insufficient accuracy
    # the cent is a practical choice of unit, and test shows that further scaling 
    # doesn't improve accuracy for most main-sequence temperaments

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
        gen, tempered_tuning_map, error_map = __optimizer_main (
            breeds, target, norm, constraint, destretch, show)
        error_map_x = norm.val_transform (error_map, target)
        error = __power_mean_norm (error_map_x)
        bias = __mean (error_map_x)
    else:
        breeds_mp, target_mp = te.breeds2mp (breeds, target)
        gen_mp, tempered_tuning_map_mp, error_map_mp = __optimizer_main (
            breeds_mp, target_mp, norm, constraint, destretch, show)
        error_map_mp_x = norm.val_transform (error_map_mp, target_mp)
        error = __power_mean_norm (error_map_mp_x)
        bias = __mean (error_map_mp_x)

        just_tuning_map = target.just_tuning_map (scalar = te.SCALAR.CENT)
        tempered_tuning_map = tempered_tuning_map_mp @ target2mp
        gen = tempered_tuning_map @ linalg.pinv (breeds)
        error_map = tempered_tuning_map - just_tuning_map

    if show:
        print (f"Generators: {gen} (¢)",
            f"Tuning map: {tempered_tuning_map} (¢)",
            f"Error map: {error_map} (¢)", sep = "\n")
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    return gen, tempered_tuning_map, error_map

def __optimizer_main (breeds, target, norm, constraint, destretch, show):
    """
    Returns the optimal generator tuning map, tempered tuning map, 
    and error map inharmonically in cents. 
    """

    just_tuning_map = target.just_tuning_map (scalar = te.SCALAR.CENT)
    breeds_x = norm.val_transform (breeds, target)
    just_tuning_map_x = norm.val_transform (just_tuning_map, target)
    if norm.order == 2 and constraint is None: #simply using lstsq for better performance
        res = linalg.lstsq (breeds_x.T, just_tuning_map_x)
        gen = res[0]
        if show: 
            print ("Euclidean tuning without constraints, solved using lstsq. ")
    else:
        gen0 = just_tuning_map[:breeds.shape[0]] #initial guess
        if constraint is None:
            cons_object = ()
        else:
            cons_basis_matrix = constraint.basis_matrix_to (target)
            cons_object = optimize.LinearConstraint ((breeds @ cons_basis_matrix).T, 
                lb = (just_tuning_map @ cons_basis_matrix).T, 
                ub = (just_tuning_map @ cons_basis_matrix).T)
        res = optimize.minimize (
            lambda gen: linalg.norm (gen @ breeds_x - just_tuning_map_x, ord = norm.order), 
            gen0, method = "COBYQA", constraints = cons_object)
        if show: 
            print (res.message)
        if res.success:
            gen = res.x
        else:
            raise ValueError ("infeasible optimization problem. ")

    if destretch is not None:
        des_monzo = te.ratio2monzo (te.as_ratio (destretch), subgroup = target)
        if (des_tempered_size := gen @ breeds @ des_monzo) == 0:
            raise ZeroDivisionError ("destretch target is in the nullspace. ")
        else:
            gen *= (just_tuning_map @ des_monzo)/des_tempered_size

    tempered_tuning_map = gen @ breeds
    error_map = tempered_tuning_map - just_tuning_map

    return gen, tempered_tuning_map, error_map

__optimiser_main = __optimizer_main
