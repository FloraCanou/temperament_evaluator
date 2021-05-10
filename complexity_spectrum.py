# © 2020-2021 Flora Canou | Version 0.4
# This work is licensed under the GNU General Public License version 3.

import numpy as np
from scipy import linalg
np.set_printoptions (suppress = True, linewidth = 256)

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

# takes a monzo, returns the ratio in [num, den] form
# doesn't check the validity of the basis
# ratio[0]: num, ratio[1]: den
def monzo2ratio (monzo, basis):
    ratio = [1, 1]
    for i in range (len (monzo)):
        if monzo[i] > 0:
            ratio[0] *= basis[i]**monzo[i]
        elif monzo[i] < 0:
            ratio[1] *= basis[i]**(-monzo[i])
    return ratio

# te weighting matrix
def weighter (subgroup):
    return np.diag (1/np.log2 (subgroup))

def find_temperamental_norm (map, monzo, subgroup, oe = True, show = True):
    if oe: #octave equivalence
        map = map[1:]
    P = linalg.pinv (map @ weighter (subgroup) @ weighter (subgroup) @ map.T)
    image = map @ monzo
    norm = np.sqrt (image.T @ P @ image)
    if show:
        ratio = monzo2ratio (monzo, subgroup)
        print (f"{ratio[0]}/{ratio[1]}\t {norm:.4f}")
    return norm

def find_spectrum (map, monzo_list, subgroup = None, oe = True):
    if subgroup == None:
        subgroup = PRIME_LIST[:map.shape[1]]
    elif map.shape[1] != len (subgroup):
        raise IndexError ("dimension does not match. ")
    spectrum = []
    for i in range (monzo_list.shape[1]):
        spectrum.append ([monzo_list[:,i], find_temperamental_norm (map, monzo_list[:,i], subgroup, oe = oe, show = False)])
    spectrum.sort (key = lambda k: k[1])
    for i in range (len (spectrum)):
        ratio = monzo2ratio (spectrum[i][0], subgroup)
        print (f"{ratio[0]}/{ratio[1]}\t{round (spectrum[i][1], 4)}")

# monzo list library
MONZO9 = np.transpose ([[2, -1, 0, 0], [-2, 0, 1, 0], [1, 1, -1, 0], [3, 0, 0, -1], [-1, -1, 0, 1], [0, 0, -1, 1], [-3, 2, 0, 0], [1, -2, 1, 0], [0, 2, 0, -1]])
MONZO11 = np.transpose ([[2, -1, 0, 0, 0], [-2, 0, 1, 0, 0], [1, 1, -1, 0, 0], [3, 0, 0, -1, 0], [-1, -1, 0, 1, 0], [0, 0, -1, 1, 0], [-3, 2, 0, 0, 0], [1, -2, 1, 0, 0], [0, 2, 0, -1, 0], \
[-3, 0, 0, 0, 1], [2, 1, 0, 0, -1], [0, -2, 0, 0, 1], [-1, 0, -1, 0, 1], [1, 0, 0, 1, -1]])
MONZO15 = np.transpose ([[2, -1, 0, 0, 0, 0], [-2, 0, 1, 0, 0, 0], [1, 1, -1, 0, 0, 0], [3, 0, 0, -1, 0, 0], [-1, -1, 0, 1, 0, 0], [0, 0, -1, 1, 0, 0], [-3, 2, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 2, 0, -1, 0, 0], \
[-3, 0, 0, 0, 1, 0], [2, 1, 0, 0, -1, 0], [0, -2, 0, 0, 1, 0], [-1, 0, -1, 0, 1, 0], [1, 0, 0, 1, -1, 0], \
[-3, 0, 0, 0, 0, 1], [-2, -1, 0, 0, 0, 1], [1, 2, 0, 0, 0, -1], [-1, 0, -1, 0, 0, 1], [1, 0, 0, 1, 0, -1], \
[4, -1, -1, 0, 0, 0], [-1, 1, 1, -1, 0, 0], [0, 1, 1, 0, -1, 0], [0, 1, 1, 0, 0, -1]])
MONZO21no11no13 = np.transpose ([[2, -1, 0, 0, 0, 0], [-2, 0, 1, 0, 0, 0], [1, 1, -1, 0, 0, 0], [3, 0, 0, -1, 0, 0], [-1, -1, 0, 1, 0, 0], [0, 0, -1, 1, 0, 0], [-3, 2, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 2, 0, -1, 0, 0], \
[4, -1, -1, 0, 0, 0], [-1, 1, 1, -1, 0, 0], [-4, 1, 0, 1, 0, 0], [-2, 1, -1, 1, 0, 0], \
[-4, 0, 0, 0, 1, 0], [3, 1, 0, 0, -1, 0], [1, 2, 0, 0, -1, 0], [2, 0, 1, 0, -1, 0], [0, -1, -1, 0, 1, 0], [-1, 0, 0, -1, 1, 0], [0, 1, 0, 1, -1, 0], \
[-4, 0, 0, 0, 0, 1], [3, 1, 0, 0, 0, -1], [-1, -2, 0, 0, 0, 1], [2, 0, 1, 0, 0, -1], [0, -1, -1, 0, 0, 1], [-1, 0, 0, -1, 0, 1], [0, 1, 0, 1, 0, -1], [0, 0, 0, 0, -1, 1]])

# example
map_history13 = np.array ([[1, 2, 0, 0, 1, 2], [0, 6, 0, -7, -2, 9], [0, 0, 1, 1, 1, 1]])
find_spectrum (map_history13, MONZO15)
