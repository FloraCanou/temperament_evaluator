# © 2020-2022 Flora Canou | Version 0.18
# This work is licensed under the GNU General Public License version 3.

import itertools, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, normalforms
from sympy import gcd
import te_common as te
import te_optimizer as te_opt
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

def map_normalize (map): #to HNF, only checks multirow mappings
    return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (map)).T).T).astype (int)) if map.shape[0] > 1 else map

class Temperament:
    def __init__ (self, map, subgroup = None):
        map, subgroup = te.subgroup_normalize (np.array (map), subgroup, axis = "row")
        self.subgroup = subgroup
        self.jip = np.log2 (self.subgroup)*te.SCALAR
        self.map = map_normalize (np.rint (map).astype (np.int))

    def weighted (self, main, wtype):
        return te.weighted (main, self.subgroup, wtype = wtype)

    def optimize (self, wtype = "tenney", order = 2,
            enforce = "custom", cons_monzo_list = None, des_monzo = None): #in cents
        if not enforce in {"custom", "d", "c", "xoc", "none"}:
            warnings.warn ("unknown enforcement type, using default (\"custom\"). ")
            enforce = "custom"

        if enforce == "d":
            des_monzo = np.transpose ([[1] + [0]*(len (self.subgroup) - 1)])
        elif enforce == "c":
            cons_monzo_list = np.transpose ([[1] + [0]*(len (self.subgroup) - 1)])
        elif enforce == "xoc":
            cons_monzo_list = np.transpose (self.weighted (np.ones (len (self.subgroup)), wtype = wtype))
        elif enforce == "none":
            des_monzo = None
            cons_monzo_list = None
        return te_opt.optimizer_main (self.map, subgroup = self.subgroup,
            wtype = wtype, order = order, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)

    def analyse (self, wtype = "tenney", order = 2,
            enforce = "custom", cons_monzo_list = None, des_monzo = None):
        if order == 2:
            order_description = "euclidean"
        elif order == np.inf:
            order_description = "chebyshevian"
        elif order == 1:
            order_description = "minkowskian"
        else:
            order_description = f"L{order}"
        if enforce == "d":
            enforce_description = "destretched"
        elif enforce == "c":
            enforce_description = "constrained"
        elif enforce == "xoc":
            enforce_description = f"{wtype} ones constrained"
        elif enforce == "none" or cons_monzo_list is None and des_monzo is None:
            enforce_description = "none"
        else:
            enforce_description = "custom"
        subgroup_string = ".".join (map (str, self.subgroup))
        print (f"\nSubgroup: {subgroup_string}",
            f"Mapping: \n{self.map}",
            f"Method: {wtype}-{order_description}. Enforcement: {enforce_description}", sep = "\n")

        gen, tuning_map = self.optimize (wtype = wtype, order = order,
            enforce = enforce, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)
        tuning_map_w = self.weighted (tuning_map, wtype = wtype)
        mistuning_map = tuning_map - self.jip
        mistuning_map_w = self.weighted (mistuning_map, wtype = wtype)
        error = linalg.norm (mistuning_map_w, ord = order) / np.sqrt (self.map.shape[1])
        bias = np.mean (mistuning_map_w)
        print (f"Mistuning map: {mistuning_map} (¢)",
            f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    optimise = optimize
    analyze = analyse

    def wedgie (self, wtype = "tenney"):
        combination_list = itertools.combinations (range (self.map.shape[1]), self.map.shape[0])
        wedgie = np.array ([linalg.det (self.weighted (self.map, wtype = wtype)[:, entry]) for entry in combination_list])
        return wedgie if wedgie[0] >= 0 else -wedgie

    def complexity (self, ntype = "breed", wtype = "tenney"):
        if not ntype in {"breed", "smith", "l2"}:
            ntype = "breed"
            warnings.warn ("unknown ntype, using default (\"breed\")")

        #standard L2 complexity
        complexity = np.sqrt (linalg.det (self.weighted (self.map, wtype = wtype) @ self.weighted (self.map, wtype = wtype).T))
        # complexity = linalg.norm (self.wedgie (wtype = wtype)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.map.shape[1]**self.map.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (self.wedgie ()))

        return complexity

    def error (self, ntype = "breed", wtype = "tenney"): #in cents
        if not ntype in {"breed", "smith", "l2"}:
            ntype = "breed"
            warnings.warn ("unknown ntype, using default (\"breed\")")

        #standard L2 error
        error = linalg.norm (
            self.weighted (self.jip, wtype)
            @ linalg.pinv (self.weighted (self.map, wtype))
            @ self.weighted (self.map, wtype)
            - (self.weighted (self.jip, wtype)))
        if ntype == "breed": #Graham Breed's RMS (default)
            error *= 1/np.sqrt (self.map.shape[1])
        elif ntype == "smith": #Gene Ward Smith's RMS
            try:
                error *= np.sqrt ((self.map.shape[0] + 1) / (self.map.shape[1] - self.map.shape[0]))
            except ZeroDivisionError:
                error = np.nan

        return error

    def badness (self, ntype = "breed", wtype = "tenney"): #in octaves
        return (self.error (ntype = ntype, wtype = wtype)
            * self.complexity (ntype = ntype, wtype = wtype)
            / te.SCALAR)

    def badness_logflat (self, ntype = "breed", wtype = "tenney"): #in octaves
        try:
            return (self.error (ntype = ntype, wtype = wtype)
                * self.complexity (ntype = ntype, wtype = wtype)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1)
                / te.SCALAR)
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", wtype = "tenney", badness_scale = 100):
        subgroup_string = ".".join (map (str, self.subgroup))
        print (f"\nSubgroup: {subgroup_string}",
            f"Mapping: \n{self.map}",
            f"Norm: {ntype}. Weighter: {wtype}", sep = "\n")

        error = self.error (ntype = ntype, wtype = wtype)
        complexity = self.complexity (ntype = ntype, wtype = wtype)
        badness = self.badness (ntype = ntype, wtype = wtype) * badness_scale
        badness_logflat = self.badness_logflat (ntype = ntype, wtype = wtype) * badness_scale
        print (f"Complexity: {complexity:.6f}",
            f"Error: {error:.6f} (¢)",
            f"Badness (simple): {badness:.6f} ({badness_scale}oct)",
            f"Badness (logflat): {badness_logflat:.6f} ({badness_scale}oct)", sep = "\n")

    def comma_basis (self, show = True):
        comma_basis_frac = Matrix (self.map).nullspace ()
        comma_basis = np.transpose ([np.squeeze (entry/gcd (tuple (entry))) for entry in comma_basis_frac])
        if show:
            subgroup_string = ".".join (map (str, self.subgroup))
            print (f"\nSubgroup: {subgroup_string}",
                f"Mapping: \n{self.map}",
                "Comma basis: ", sep = "\n")
            for i in range (comma_basis.shape[1]):
                monzo = comma_basis[:, i]
                ratio = te.monzo2ratio (monzo, self.subgroup)
                monzo_str = "[" + " ".join (map (str, np.trim_zeros (monzo, trim = "b"))) + ">"
                if ratio[0] < 10e7:
                    print (monzo_str, f"({ratio[0]}/{ratio[1]})")
                else:
                    print (monzo_str)
        return comma_basis
