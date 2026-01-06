# © 2020-2026 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import math, warnings
import numpy as np
from scipy import linalg
from . import te_common as te
from . import te_temperament_measures as te_tm

class TemperamentLattice (te_tm.Temperament):
    def temperamental_norm (self, monzo, norm = te.Norm (), oe = False, show = True):
        """Takes a monzo and returns its temperamental norm."""
        mapping_copy = self.mapping[1:] if oe else self.mapping # formal-octave equivalence
        mapping_x = norm.val_transform (mapping_copy, self.subgroup)
        projection_x = linalg.pinv (mapping_x) @ mapping_x
        monzo_x = norm.interval_transform (monzo, self.subgroup)
        interval_temperamental_norm = np.sqrt (monzo_x.T @ projection_x @ monzo_x)
        if show:
            ratio = te.monzo2ratio (monzo, self.subgroup)
            print (f"{ratio}\t {interval_temperamental_norm}")
        return interval_temperamental_norm

    def temperamental_complexity_spectrum (self, monzos, norm = te.Norm (), oe = True):
        """Takes a monzo list and displays the temperamental norms."""
        monzos, _ = te.setup (monzos, self.subgroup, axis = te.AXIS.COL)
        spectrum = [[monzos[:, i], self.temperamental_norm (
            monzos[:, i], norm, oe, show = False
            )] for i in range (monzos.shape[1])]
        spectrum.sort (key = lambda k: k[1])

        if oe:
            print ("\nOctave-equivalent complexity spectrum: ")
        else:
            print ("\nComplexity spectrum: ")
        for entry in spectrum:
            ratio = te.monzo2ratio (entry[0], self.subgroup)
            print (f"{ratio}\t{entry[1]:.3f}")

def diamond_monzos_gen (limit, eq = None, excl = [], comp = True, sort = None):
    """
    Enter a limit and an equave, returns an array of monzos
    for the corresponding tonality diamond. 
    """
    excl = te.as_list (excl)
    ratio_list = []
    for num in range (1, limit + 1):
        if num in excl or eq and num % eq == 0:
            continue
        for den in range (1, num + 1):
            if (den in excl or eq and den % eq == 0 
                    or math.gcd (num, den) != 1):
                continue
            ratio = te.Ratio (num, den)
            if eq: 
                ratio_list.append (ratio.eq_reduce (eq))
                if comp: 
                    ratio_list.append (ratio.eq_reduce (eq).eq_complement (eq))
            else: 
                ratio_list.append (ratio)
                
    if sort == "size":
        ratio_list.sort (key = lambda c: c.value ())
    return te.column_stack_pad ([te.ratio2monzo (entry) for entry in ratio_list])
