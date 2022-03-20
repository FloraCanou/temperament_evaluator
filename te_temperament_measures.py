# © 2020-2022 Flora Canou | Version 0.12
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, normalforms
import itertools, warnings
import tuning_optimizer
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SCALAR = 1200 #could be in octave, but for precision reason

def map_normalize (map): #to HNF, only checks multirow mappings
    return np.flip (np.array (normalforms.hermite_normal_form (Matrix (np.flip (map)).T).T).astype (int)) if map.shape[0] > 1 else map

class Temperament:
    def __init__ (self, map, subgroup = None):
        self.map = map_normalize (np.rint (map).astype (np.int))
        self.subgroup = PRIME_LIST[:self.map.shape[1]] if subgroup is None else subgroup
        self.jip = np.log2 (self.subgroup)*SCALAR

    def weighted (self, matrix, wtype):
        return tuning_optimizer.weighted (matrix, self.subgroup, wtype = wtype)

    def optimize (self, wtype = "tenney", order = 2, enforce = "custom", cons_monzo_list = None, stretch_monzo = None): #in cents
        if not enforce in {"custom", "po", "c", "xoc", "none"}:
            enforce = "custom"
            warning.warn ("unknown enforcement type, using default (\"custom\")")

        if enforce == "po":
            stretch_monzo = np.transpose ([1] + [0]*(len (self.subgroup) - 1))
        elif enforce == "c":
            cons_monzo_list = np.transpose ([1] + [0]*(len (self.subgroup) - 1))
        elif enforce == "xoc":
            cons_monzo_list = np.transpose (self.weighted (np.ones (len (self.subgroup)), wtype = wtype))
        elif enforce == "none":
            stretch_monzo = None
            cons_monzo_list = None
        return tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, wtype = wtype, order = order, cons_monzo_list = cons_monzo_list, stretch_monzo = stretch_monzo)

    def analyse (self, wtype = "tenney", order = 2, enforce = "custom", cons_monzo_list = None, stretch_monzo = None): #in octaves
        if order == 2:
            order_description = "euclidean"
        elif order == np.inf:
            order_description = "chebyshevian"
        elif order == 1:
            order_description = "minkowskian"
        else:
            order_description = f"L{order}"
        if enforce == "po":
            enforce_description = "stretched"
        elif enforce == "c":
            enforce_description = "constrained"
        elif enforce == "xoc":
            enforce_description = f"{wtype} ones constrained"
        elif enforce == "none" or cons_monzo_list is None and stretch_monzo is None:
            enforce_description = "none"
        else:
            enforce_description = "custom"
        print (f"\nMapping: \n{self.map}", f"Method: {wtype}-{order_description}. Enforcement: {enforce_description}", sep = "\n")

        gen, tuning_map = self.optimize (wtype = wtype, order = order, enforce = enforce, cons_monzo_list = cons_monzo_list, stretch_monzo = stretch_monzo)
        tuning_map_w = self.weighted (tuning_map, wtype = wtype)
        mistuning_map = tuning_map - self.jip
        mistuning_map_w = self.weighted (mistuning_map, wtype = wtype)
        error = linalg.norm (mistuning_map_w, ord = order) / np.sqrt (self.map.shape[1])
        bias = np.mean (mistuning_map_w)
        print (f"Mistuning map: {mistuning_map} (¢)", f"Tuning error: {error:.6f} (¢)", f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    optimise = optimize
    analyze = analyse

    def wedgie (self, wtype = "tenney"):
        combination_list = itertools.combinations (range (self.map.shape[1]), self.map.shape[0])
        wedgie = [linalg.det (self.weighted (self.map, wtype = wtype)[:,entry]) for entry in combination_list]
        return np.array (wedgie) if wedgie[0] >= 0 else -np.array (wedgie)

    def complexity (self, ntype = "breed", wtype = "tenney"):
        if not ntype in {"breed", "smith", "l2"}:
            ntype = "breed"
            warnings.warn ("unknown ntype, using default (\"breed\")")

        if ntype == "breed": #Graham Breed's RMS (default)
            complexity = np.sqrt (linalg.det (self.weighted (self.map, wtype = wtype) @ self.weighted (self.map, wtype = wtype).T / self.map.shape[1]))
            # complexity = linalg.norm (self.wedgie (wtype = wtype)) / np.sqrt (self.map.shape[1]**self.map.shape[0]) #same
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity = np.sqrt (linalg.det (self.weighted (self.map, wtype = wtype) @ self.weighted (self.map, wtype = wtype).T) / len (self.wedgie ()))
            # complexity = linalg.norm (self.wedgie (wtype = wtype)) / np.sqrt (len (self.wedgie ())) #same
        elif ntype == "l2": #standard L2
            complexity = np.sqrt (linalg.det (self.weighted (self.map, wtype = wtype) @ self.weighted (self.map, wtype = wtype).T))
            # complexity = linalg.norm (self.wedgie (wtype = wtype)) #same
        return complexity

    def error (self, ntype = "breed", wtype = "tenney"): #in cents
        if not ntype in {"breed", "smith", "l2"}:
            ntype = "breed"
            warnings.warn ("unknown ntype, using default (\"breed\")")

        if ntype == "breed": #Graham Breed's RMS (default)
            error = linalg.norm (self.weighted (self.jip, wtype = wtype) @ (linalg.pinv (self.weighted (self.map, wtype = wtype)) @ self.weighted (self.map, wtype = wtype) - np.eye (self.map.shape[1]))) / np.sqrt (self.map.shape[1])
        elif ntype == "smith": #Gene Ward Smith's RMS
            error = linalg.norm (self.weighted (self.jip, wtype = wtype) @ (linalg.pinv (self.weighted (self.map, wtype = wtype)) @ self.weighted (self.map, wtype = wtype) - np.eye (self.map.shape[1]))) * np.sqrt ((self.map.shape[0] + 1) / (self.map.shape[1] - self.map.shape[0]))
        elif ntype == "l2": #standard L2
            error = linalg.norm (self.weighted (self.jip, wtype = wtype) @ (linalg.pinv (self.weighted (self.map, wtype = wtype)) @ self.weighted (self.map, wtype = wtype) - np.eye (self.map.shape[1])))
        return error

    def badness (self, ntype = "breed", wtype = "tenney"): #in octaves
        return self.error (ntype = ntype, wtype = wtype) * self.complexity (ntype = ntype, wtype = wtype) / SCALAR

    def badness_logflat (self, ntype = "breed", wtype = "tenney"): #in octaves
        return self.error (ntype = ntype, wtype = wtype) * self.complexity (ntype = ntype, wtype = wtype)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1) / SCALAR

    def temperament_measures (self, ntype = "breed", wtype = "tenney", badness_scale = 100):
        print (f"\nMapping: \n{self.map}", f"Norm: {ntype}. Weighter: {wtype}", sep = "\n")
        error = self.error (ntype = ntype, wtype = wtype)
        complexity = self.complexity (ntype = ntype, wtype = wtype)
        badness = self.badness (ntype = ntype, wtype = wtype) * badness_scale
        badness_logflat = self.badness_logflat (ntype = ntype, wtype = wtype) * badness_scale
        print (f"Complexity: {complexity:.6f}", f"Error: {error:.6f} (¢)", f"Badness (simple): {badness:.6f} ({badness_scale}oct)", f"Badness (logflat): {badness_logflat:.6f} ({badness_scale}oct)", sep = "\n")
