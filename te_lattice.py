# © 2020-2023 Flora Canou | Version 0.27.2
# This work is licensed under the GNU General Public License version 3.

import math, warnings
import numpy as np
from scipy import linalg
import te_common as te
import te_temperament_measures as te_tm
np.set_printoptions (suppress = True, linewidth = 256)

class TemperamentLattice (te_tm.Temperament):
    def find_temperamental_norm (self, monzo, norm = te.Norm (), oe = False, show = True):
        mapping_copy = self.mapping[1:] if oe else self.mapping # octave equivalence
        mapping_x = norm.tuning_x (mapping_copy, self.subgroup)
        projection_x = linalg.pinv (mapping_x) @ mapping_x
        monzo_x = norm.interval_x (monzo, self.subgroup)
        interval_temperamental_norm = np.sqrt (monzo_x.T @ projection_x @ monzo_x)
        if show:
            ratio = te.monzo2ratio (monzo, self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t {interval_temperamental_norm}")
        return interval_temperamental_norm

    def find_complexity_spectrum (self, monzos, norm = te.Norm (), oe = True):
        monzos, _ = te.setup (monzos, self.subgroup, axis = te.AXIS.COL)
        spectrum = [[monzos[:, i], self.find_temperamental_norm (
            monzos[:, i], norm, oe, show = False
            )] for i in range (monzos.shape[1])]
        spectrum.sort (key = lambda k: k[1])
        print ("\nComplexity spectrum: ")
        for entry in spectrum:
            ratio = te.monzo2ratio (entry[0], self.subgroup)
            print (f"{ratio[0]}/{ratio[1]}\t{entry[1]:.4f}")

    find_spectrum = find_complexity_spectrum

def ratio_octave_reduce (ratio): #NOTE: "oct" is a reserved word
    """Octave-reduces the ratio in [num, den] form."""
    num_oct = np.floor (np.log2 (ratio[0]/ratio[1])) 
    if num_oct > 0:
        ratio[1] *= round (2**num_oct)
    elif num_oct < 0:
        ratio[0] *= round (2**(-num_oct))
    return ratio

def odd_limit_monzos_gen (odd_limit, excl = [], sort = None):
    """Enter an odd limit, returns the array of monzos."""
    ratio_list = []
    for num in range (1, odd_limit + 1, 2):
        if num in te.as_list (excl):
            continue
        for den in range (1, num + 1, 2):
            if (den in te.as_list (excl)) or math.gcd (num, den) != 1:
                continue
            ratio_list.append (ratio_octave_reduce ([num, den]))
    if sort == "size":
        ratio_list.sort (key = lambda k: k[0]/k[1])
    return te.column_stack_pad ([te.ratio2monzo (entry) for entry in ratio_list])

def integer_limit_monzos_gen (integer_limit, excl = [], sort = None):
    """Enter an integer limit, returns the array of monzos."""
    ratio_list = []
    for num in range (1, integer_limit + 1):
        if num in te.as_list (excl):
            continue
        for den in range (1, num + 1):
            if (den in te.as_list (excl)) or math.gcd (num, den) != 1:
                continue
            ratio_list.append (ratio_octave_reduce ([num, den]))
    if sort == "size":
        ratio_list.sort (key = lambda c: c.value ())
    return te.column_stack_pad ([te.ratio2monzo (entry) for entry in ratio_list])
