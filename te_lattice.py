# © 2020-2023 Flora Canou | Version 0.25.0
# This work is licensed under the GNU General Public License version 3.

import math, warnings
import numpy as np
from scipy import linalg
import te_common as te
import te_temperament_measures as te_tm
np.set_printoptions (suppress = True, linewidth = 256)

class TemperamentLattice (te_tm.Temperament):
    def find_temperamental_norm (self, monzo, norm = te.Norm (), oe = False, show = True, 
            *, wtype = None, wamount = None, skew = None, order = None):

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        # octave equivalence
        map_copy = self.map[1:] if oe else self.map
        map_wx = self.weightskewed (map_copy, norm)
        projection_wx = linalg.pinv (map_wx) @ map_wx
        interval_temperamental_norm = np.sqrt (monzo.T @ projection_wx @ monzo)
        if show:
            ratio = te.monzo2ratio (monzo, self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t {interval_temperamental_norm}")
        return interval_temperamental_norm

    def find_spectrum (self, monzo_list, norm = te.Norm (), oe = True, 
        *, wtype = None, wamount = None, skew = None, order = None):
        monzo_list, _ = te.get_subgroup (monzo_list, self.subgroup, axis = te.COL)

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        spectrum = [[monzo_list[:, i], self.find_temperamental_norm (
            monzo_list[:, i], norm, oe, show = False
            )] for i in range (monzo_list.shape[1])]
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
