# © 2020-2023 Flora Canou | Version 0.26.4
# This work is licensed under the GNU General Public License version 3.

import math, warnings
import numpy as np
from scipy import linalg
import te_common as te
import te_temperament_measures as te_tm
np.set_printoptions (suppress = True, linewidth = 256)

class TemperamentLattice (te_tm.Temperament):
    def find_temperamental_norm (self, monzo, norm = te.Norm (), oe = False, show = True):
        vals_copy = self.vals[1:] if oe else self.vals # octave equivalence
        vals_x = self.weightskewed (vals_copy, norm)
        projection_x = linalg.pinv (vals_x) @ vals_x
        monzo_x = linalg.pinv (norm._Norm__get_weight (self.subgroup) @ norm._Norm__get_skew (self.subgroup)) @ monzo
        interval_temperamental_norm = np.sqrt (monzo_x.T @ projection_x @ monzo_x)
        if show:
            ratio = te.monzo2ratio (monzo, self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t {interval_temperamental_norm}")
        return interval_temperamental_norm

    def find_complexity_spectrum (self, monzo_list, norm = te.Norm (), oe = True):
        monzo_list, _ = te.get_subgroup (monzo_list, self.subgroup, axis = te.AXIS.COL)
        spectrum = [[monzo_list[:, i], self.find_temperamental_norm (
            monzo_list[:, i], norm, oe, show = False
            )] for i in range (monzo_list.shape[1])]
        spectrum.sort (key = lambda k: k[1])
        print ("\nComplexity spectrum: ")
        for entry in spectrum:
            ratio = te.monzo2ratio (entry[0], self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t{entry[1]:.4f}")

    find_spectrum = find_complexity_spectrum

# octave-reduce the ratio in [num, den] form
def ratio_8ve_reduction (ratio): #NOTE: "oct" is a reserved word
    num_oct = math.floor (math.log (ratio[0]/ratio[1], 2)) 
    if num_oct > 0:
        ratio[1] *= 2**num_oct
    elif num_oct < 0:
        ratio[0] *= 2**(-num_oct)
    return ratio

# enter an odd limit, returns the monzo list
def odd_limit_monzo_list_gen (odd_limit, excl = [], sort = None):
    subgroup = list (filter (lambda q: q <= odd_limit, te.PRIME_LIST))
    ratio_list = []
    for num in range (1, odd_limit + 1, 2):
        if num in te.as_list (excl):
            continue
        for den in range (1, odd_limit + 1, 2):
            if (den in te.as_list (excl)) or math.gcd (num, den) != 1:
                continue
            ratio_list.append (ratio_8ve_reduction ([num, den]))
    if sort == "size":
        ratio_list.sort (key = lambda k: k[0]/k[1])
    return np.transpose ([te.ratio2monzo (entry, subgroup) for entry in ratio_list])
