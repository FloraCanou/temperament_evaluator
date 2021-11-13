# © 2020-2021 Flora Canou | Version 0.7
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from scipy import linalg
import itertools
import tuning_optimizer
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SCALAR = 1200 #could be in octave, but for precision reason

def map_normalize (map):
    #todo: add something
    return map

class Temperament:
    def __init__ (self, map, subgroup = []):
        self.map = map_normalize (np.array (map))
        self.subgroup = PRIME_LIST[:self.map.shape[1]] if len (subgroup) == 0 else subgroup
        self.jip = np.log2 (self.subgroup)*SCALAR
        tenney_weighter = np.diag (1/np.log2 (self.subgroup))
        self.weight = tenney_weighter

    def weighted (self, matrix):
        return matrix @ self.weight

    def unweighted (self, matrix):
        return matrix @ linalg.inv (self.weight)

    def optimize (self, type = "custom", order = 2, cons_monzo_list = np.array ([]), stretch_monzo = np.array ([])): #in cents
        if not type in {"custom", "te", "pote", "cte", "top", "potop", "ctop"}:
            print ("Type not recognized, using default")
            type = "custom"

        if type == "custom":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, order = order, cons_monzo_list = cons_monzo_list, stretch_monzo = stretch_monzo)
        elif type == "te":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup)
        elif type == "pote":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, stretch_monzo = np.transpose ([1] + [0]*(len (self.subgroup) - 1)))
        elif type == "cte":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, cons_monzo_list = np.transpose ([1] + [0]*(len (self.subgroup) - 1)))
        elif type == "top":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, order = np.inf)
        elif type == "potop":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, order = np.inf, stretch_monzo = np.transpose ([1] + [0]*(len (self.subgroup) - 1)))
        elif type == "ctop":
            gen = tuning_optimizer.optimizer_main (self.map, subgroup = self.subgroup, order = np.inf, cons_monzo_list = np.transpose ([1] + [0]*(len (self.subgroup) - 1)))
        return gen

    def analyse (self, type = "custom", order = 2, cons_monzo_list = np.array ([]), stretch_monzo = np.array ([])): #in octaves
        print (f"\nMapping: \n{self.map}", f"Type: {type}", sep = "\n")
        gen = self.optimize (type = type, order = order, cons_monzo_list = cons_monzo_list, stretch_monzo = stretch_monzo)
        tuning_map = gen @ self.map
        tuning_map_w = self.weighted (tuning_map)
        mistuning_map = tuning_map - self.jip
        mistuning_map_w = self.weighted (mistuning_map)
        error = linalg.norm (mistuning_map_w, ord = order) / np.sqrt (self.map.shape[1])
        bias = np.mean (mistuning_map_w)
        print (f"Mistuning map: {mistuning_map} (¢)", f"Tuning error: {error:.6f} (¢)", f"Tuning bias: {bias:.6f} (¢)", sep = "\n")

    def wedgie (self, weighted = False):
        combination_list = list (itertools.combinations (range (self.map.shape[1]), self.map.shape[0]))
        wedgie = []
        for entry in combination_list:
            if weighted:
                wedgie.append (linalg.det (self.weighted (self.map)[:,entry]))
            else:
                wedgie.append (int (linalg.det (self.map[:,entry])))
        return np.array (wedgie) if wedgie[0] >= 0 else -np.array (wedgie)

    def complexity (self, type = "rmsgraham"):
        if not type in {"rmsgraham", "rmsgene", "l2"}:
            type = "rmsgraham"

        if type == "rmsgraham": #Graham Breed's RMS (default)
            complexity = np.sqrt (linalg.det (self.weighted (self.map) @ self.weighted (self.map).T / self.map.shape[1]))
            # complexity = linalg.norm (self.wedgie (weighted = True)) / np.sqrt (self.map.shape[1]**self.map.shape[0]) #same
        elif type == "rmsgene": #Gene Ward Smith's RMS
            complexity = np.sqrt (linalg.det (self.weighted (self.map) @ self.weighted (self.map).T) / len (self.wedgie ()))
            # complexity = linalg.norm (self.wedgie (weighted = True)) / np.sqrt (len (self.wedgie ())) #same
        elif type == "l2": #standard L2
            complexity = np.sqrt (linalg.det (self.weighted (self.map) @ self.weighted (self.map).T))
            # complexity = linalg.norm (self.wedgie (weighted = True)) #same
        return complexity

    def error (self, type = "rmsgraham"): #in cents
        if not type in {"rmsgraham", "rmsgene", "l2"}:
            type = "rmsgraham"

        if type == "rmsgraham": #Graham Breed's RMS (default)
            error = linalg.norm (self.weighted (self.jip) @ (linalg.pinv (self.weighted (self.map)) @ self.weighted (self.map) - np.eye (self.map.shape[1]))) / np.sqrt (self.map.shape[1])
        elif type == "rmsgene": #Gene Ward Smith's RMS
            error = linalg.norm (self.weighted (self.jip) @ (linalg.pinv (self.weighted (self.map)) @ self.weighted (self.map) - np.eye (self.map.shape[1]))) * np.sqrt ((self.map.shape[0] + 1) / (self.map.shape[1] - self.map.shape[0]))
        elif type == "l2": #standard L2
            error = linalg.norm (self.weighted (self.jip) @ (linalg.pinv (self.weighted (self.map)) @ self.weighted (self.map) - np.eye (self.map.shape[1])))
        return error

    def badness (self, type = "rmsgraham"): #in octaves
        return self.error (type = type) * self.complexity (type = type) / SCALAR

    def badness_logflat (self, type = "rmsgraham"): #in octaves
        return self.error (type = type) * self.complexity (type = type)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1) / SCALAR

    def temperament_measures (self, type = "rmsgraham", badness_scale = 100):
        print (f"\nMapping: \n{self.map}", f"Type: {type}", sep = "\n")
        error = self.error (type = type)
        complexity = self.complexity (type = type)
        badness = self.badness (type = type) * badness_scale
        badness_logflat = self.badness_logflat (type = type) * badness_scale
        print (f"Complexity: {complexity:.6f}", f"Error: {error:.6f} (¢)", f"Badness (simple): {badness:.6f} ({badness_scale}oct)", f"Badness (logflat): {badness_logflat:.6f} ({badness_scale}oct)", sep = "\n")
