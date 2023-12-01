# © 2020-2023 Flora Canou | Version 1.0.0
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
        vals_x = norm.tuning_x (vals_copy, self.subgroup)
        projection_x = linalg.pinv (vals_x) @ vals_x
        monzo_x = norm.interval_x (monzo, self.subgroup)
        interval_temperamental_norm = np.sqrt (monzo_x.T @ projection_x @ monzo_x)
        if show:
            ratio = te.monzo2ratio (monzo, self.subgroup)
            print (f"{ratio}\t {interval_temperamental_norm}")
        return interval_temperamental_norm

    def find_complexity_spectrum (self, monzo_list, norm = te.Norm (), oe = True):
        monzo_list, _ = te.setup (monzo_list, self.subgroup, axis = te.AXIS.COL)
        spectrum = [[monzo_list[:, i], self.find_temperamental_norm (
            monzo_list[:, i], norm, oe, show = False
            )] for i in range (monzo_list.shape[1])]
        spectrum.sort (key = lambda k: k[1])
        print ("\nComplexity spectrum: ")
        for entry in spectrum:
            ratio = te.monzo2ratio (entry[0], self.subgroup)
            print (f"{ratio}\t{entry[1]:.4f}")

    find_spectrum = find_complexity_spectrum

# enter an odd limit, returns the monzo list
def odd_limit_monzo_list_gen (odd_limit, excl = [], sort = None):
    ratio_list = []
    for num in range (1, odd_limit + 1, 2):
        if num in te.as_list (excl):
            continue
        for den in range (1, odd_limit + 1, 2):
            if (den in te.as_list (excl)) or math.gcd (num, den) != 1:
                continue
            ratio_list.append (te.Ratio (num, den).octave_reduce ())
    if sort == "size":
        ratio_list.sort (key = lambda c: c.value ())
    return te.column_stack_pad ([te.ratio2monzo (entry) for entry in ratio_list])
