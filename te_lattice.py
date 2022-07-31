# © 2020-2022 Flora Canou | Version 0.21.0
# This work is licensed under the GNU General Public License version 3.

import math
import numpy as np
from scipy import linalg
import te_common as te
import te_temperament_measures as te_tm
np.set_printoptions (suppress = True, linewidth = 256)

class TemperamentLattice (te_tm.Temperament):
    def find_temperamental_norm (self, monzo, wtype = "tenney", skew = 0, oe = False, show = True):
        # octave equivalence
        map_copy = self.map[1:] if oe else self.map
        projection_wx = linalg.pinv (self.weightskewed (map_copy, wtype, skew)) @ self.weightskewed (map_copy, wtype, skew)
        norm = np.sqrt (monzo.T @ projection_wx @ monzo)
        if show:
            ratio = te.monzo2ratio (monzo, self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t {norm}")
        return norm

    def find_spectrum (self, monzo_list, wtype = "tenney", skew = 0, oe = True):
        monzo_list, _ = te.get_subgroup (monzo_list, self.subgroup, axis = te.COL)

        spectrum = [[monzo_list[:, i], self.find_temperamental_norm (
            monzo_list[:, i], wtype, skew, oe = oe, show = False)] for i in range (monzo_list.shape[1])]
        spectrum.sort (key = lambda k: k[1])
        print ("\nComplexity spectrum: ")
        for entry in spectrum:
            ratio = te.monzo2ratio (entry[0], self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t{entry[1]:.4f}")

# octave-reduce the ratio in [num, den] form
def ratio_8ve_reduction (ratio):
    oct = math.floor (math.log (ratio[0]/ratio[1], 2))
    if oct > 0:
        ratio[1] *= 2**oct
    elif oct < 0:
        ratio[0] *= 2**(-oct)
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
