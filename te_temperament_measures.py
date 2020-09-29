import numpy as np
from scipy import linalg
import itertools
np.set_printoptions (suppress = True, linewidth = 256)

def map_normalize (map):
    #todo: add something
    return map

class Temperament:
    def __init__ (self, map, subgroup):
        self.map = map_normalize (np.array (map))
        self.subgroup = subgroup
        self.jip = np.log2 (subgroup)
        weight_te = np.eye (self.map.shape[1])
        for i in range (self.map.shape[1]):
            try:
                weight_te[i][i] = 1/np.log2 (subgroup[i])
            except ZeroDivisionError:
                continue
        self.weight = weight_te

    def weighted (self, matrix):
        return matrix @ self.weight

    def unweighted (self, matrix):
        return matrix @ linalg.pinv (self.weight)

    def gen (self, pote = False): #in octaves
        gen = linalg.lstsq (self.weighted (self.map).T, self.weighted (self.jip))[0]
        return gen if not pote else gen / (gen @ self.map)[0]

    def tuning_map (self, pote = False, unweighted = False): #in octaves
        return self.gen (pote = pote) @ (self.weighted (self.map) if not unweighted else self.map)

    def mistuning_map (self, pote = False, unweighted = False): #in octaves
        return self.tuning_map (pote = pote, unweighted = unweighted) - (self.weighted (self.jip) if not unweighted else self.jip)

    def error (self, type = "rmsgraham", pote = False):
        error_l2 = linalg.norm (self.mistuning_map (pote = pote))
        if type == "l2": #standard L2
            return error_l2
        elif type == "rmsgraham": #RMS normalized by the rank
            return error_l2 / np.sqrt (self.map.shape[1])
        elif type == "rmsgene": #RMS
            try:
                return error_l2 * np.sqrt ((self.map.shape[0] + 1) / (self.map.shape[1] - self.map.shape[0]))
            except ZeroDivisionError:
                return np.NAN
        elif type == "l1" or type == "top": #bonus: L1/TOP error
            return linalg.norm (self.mistuning_map (type = type, pote = pote), ord = 1)

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
        complexity_l2 = linalg.norm (self.wedgie (weighted = True))
        #complexity_l2 = np.sqrt (linalg.det (self.weighted (self.map) @ self.weighted (self.map).T)) #same
        if type == "l2": #standard L2 complexity
            return complexity_l2
        elif type == "rmsgraham": #RMS normalized by the rank
            return complexity_l2 / np.sqrt (self.map.shape[1]**self.map.shape[0])
            #return np.sqrt (linalg.det (self.weighted (self.map) @ self.weighted (self.map).T / self.map.shape[1])) #same
        elif type == "rmsgene": #RMS
            return complexity_l2 / np.sqrt (len (self.wedgie ()))
            #return np.sqrt (linalg.det (self.weighted (self.map) @ self.weighted (self.map).T) / len (self.wedgie ())) #same
        elif type == "l1" or type == "top": #bonus: L1/TOP complexity
            return linalg.norm (self.wedgie (weighted = True), ord = 1)

    def simple_badness (self, type = "rmsgraham"):
        return self.complexity (type = type) * self.error (type = type)

    def logflat_badness (self, type = "rmsgraham"):
        try:
            return self.simple_badness (type = type) * self.complexity (type = type)**(self.map.shape[0]/(self.map.shape[1] - self.map.shape[0]))
        except ZeroDivisionError:
            return np.NAN

    def show_input (self, pote, measure):
        print (f"\nMap: \n{self.map}")
        if pote:
            print ("====== In POTE ======")

    def measure2scale (self, measure):
        if measure == "cent":
            return 1200
        else:
            return 1

    def show_tuning_map (self, show_input = True, pote = False, measure = "cent"):
        if show_input:
            self.show_input (pote, measure)
        scale = self.measure2scale (measure)
        print (f"Generators: {scale*self.gen (pote = pote)}")
        print (f"Tuning map: {scale*self.tuning_map (pote = pote)}")
        print (f"Mistuning map: {scale*self.mistuning_map (pote = pote)}")

    def show_temperament_measures (self, show_input = True, pote = False, type = "rmsgraham", measure = "cent", badness_scale = 1000):
        if show_input:
            self.show_input (pote, measure)
        scale = self.measure2scale (measure)
        print (f"Error: {scale*self.error (type = type, pote = pote):.6f} ({type})")
        print (f"Complexity: {self.complexity (type = type):.6f} ({type})")
        print (f"Simple badness: {badness_scale*self.simple_badness (type = type):.6f} ({badness_scale}x {type})")

    def show_all (self, show_input = True, pote = False, wedgie_weighted = False, type = "rmsgraham", measure = "cent", badness_scale = 1000):
        if show_input:
            self.show_input (pote, measure)
        self.show_tuning_map (show_input = False, pote = pote, measure = measure)
        self.show_temperament_measures (show_input = False, pote = pote, type = type, measure = measure, badness_scale = badness_scale)
        print (f"Logflat badness: {badness_scale*self.logflat_badness (type = type):.6f} ({badness_scale}x {type})")
        print (f"Wedgie: {self.wedgie (weighted = wedgie_weighted)}")

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

#Et construction function
def logn (m, n):
    return np.log (m)/np.log (n)

def et_construct (n, ed_ratio, subgroup, alt_val = 0):
    val = np.rint (n*logn (subgroup, ed_ratio)).astype (int, copy = False) + alt_val
    return Temperament ([val], subgroup)
