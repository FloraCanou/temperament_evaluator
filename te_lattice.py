# © 2020-2022 Flora Canou | Version 0.15
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from scipy import linalg
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256)

# takes a monzo, returns the ratio in [num, den] form
# doesn't validate the basis
# ratio[0]: num, ratio[1]: den
def monzo2ratio (monzo, subgroup = None):
    monzo, subgroup = te.subgroup_normalize (monzo, subgroup, axis = "vec")
    ratio = [1, 1]
    for i in range (len (monzo)):
        if monzo[i] > 0:
            ratio[0] *= subgroup[i]**monzo[i]
        elif monzo[i] < 0:
            ratio[1] *= subgroup[i]**(-monzo[i])
    return ratio

def find_temperamental_norm (map, monzo, subgroup, wtype = "tenney", oe = True, show = True):
    if oe: #octave equivalence
        map = map[1:]
    projection_w = linalg.pinv (te.weighted (map, subgroup, wtype = wtype) @ te.weighted (map, subgroup, wtype = wtype).T)
    image = map @ monzo
    norm = np.sqrt (image.T @ projection_w @ image)
    if show:
        ratio = monzo2ratio (monzo, subgroup)
        print (f"{ratio[0]}/{ratio[1]}\t {norm:.4f}")
    return norm

def find_spectrum (map, monzo_list, subgroup = None, wtype = "tenney", oe = True):
    map, subgroup = te.subgroup_normalize (map, subgroup, axis = "row")

    spectrum = [[monzo_list[:, i], find_temperamental_norm (map, monzo_list[:, i], subgroup, wtype = wtype, oe = oe, show = False)] for i in range (monzo_list.shape[1])]
    spectrum.sort (key = lambda k: k[1])
    for entry in spectrum:
        ratio = monzo2ratio (entry[0], subgroup)
        print (f"{ratio[0]}/{ratio[1]}\t{entry[1]:.4f}")

# monzo list library
MONZO9 = np.transpose ([[2, -1, 0, 0], [-2, 0, 1, 0], [1, 1, -1, 0], [3, 0, 0, -1], [-1, -1, 0, 1], [0, 0, -1, 1], [-3, 2, 0, 0], [1, -2, 1, 0], [0, 2, 0, -1]])
MONZO11 = np.transpose ([[2, -1, 0, 0, 0], [-2, 0, 1, 0, 0], [1, 1, -1, 0, 0], [3, 0, 0, -1, 0], [-1, -1, 0, 1, 0], [0, 0, -1, 1, 0], [-3, 2, 0, 0, 0], [1, -2, 1, 0, 0], [0, 2, 0, -1, 0], \
[-3, 0, 0, 0, 1], [2, 1, 0, 0, -1], [0, -2, 0, 0, 1], [-1, 0, -1, 0, 1], [1, 0, 0, 1, -1]])
MONZO15 = np.transpose ([[2, -1, 0, 0, 0, 0], [-2, 0, 1, 0, 0, 0], [1, 1, -1, 0, 0, 0], [3, 0, 0, -1, 0, 0], [-1, -1, 0, 1, 0, 0], [0, 0, -1, 1, 0, 0], [-3, 2, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 2, 0, -1, 0, 0], \
[-3, 0, 0, 0, 1, 0], [2, 1, 0, 0, -1, 0], [0, -2, 0, 0, 1, 0], [-1, 0, -1, 0, 1, 0], [1, 0, 0, 1, -1, 0], \
[4, 0, 0, 0, 0, -1], [-2, -1, 0, 0, 0, 1], [1, 2, 0, 0, 0, -1], [-1, 0, -1, 0, 0, 1], [1, 0, 0, 1, 0, -1], [0, 0, 0, 0, -1, 1], \
[4, -1, -1, 0, 0, 0], [-1, 1, 1, -1, 0, 0], [0, 1, 1, 0, -1, 0], [0, 1, 1, 0, 0, -1]])
MONZO21no11no13 = np.transpose ([[2, -1, 0, 0, 0, 0], [-2, 0, 1, 0, 0, 0], [1, 1, -1, 0, 0, 0], [3, 0, 0, -1, 0, 0], [-1, -1, 0, 1, 0, 0], [0, 0, -1, 1, 0, 0], [-3, 2, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 2, 0, -1, 0, 0], \
[4, -1, -1, 0, 0, 0], [-1, 1, 1, -1, 0, 0], [-4, 1, 0, 1, 0, 0], [-2, 1, -1, 1, 0, 0], \
[-4, 0, 0, 0, 1, 0], [3, 1, 0, 0, -1, 0], [1, 2, 0, 0, -1, 0], [2, 0, 1, 0, -1, 0], [0, -1, -1, 0, 1, 0], [-1, 0, 0, -1, 1, 0], [0, 1, 0, 1, -1, 0], \
[-4, 0, 0, 0, 0, 1], [3, 1, 0, 0, 0, -1], [-1, -2, 0, 0, 0, 1], [2, 0, 1, 0, 0, -1], [0, -1, -1, 0, 0, 1], [-1, 0, 0, -1, 0, 1], [0, 1, 0, 1, 0, -1], [0, 0, 0, 0, -1, 1]])

# example
map_history13 = np.array ([[1, 2, 0, 0, 1, 2], [0, 6, 0, -7, -2, 9], [0, 0, 1, 1, 1, 1]])
find_spectrum (map_history13, MONZO15)
