# Copyright 2020 Flora Canou
# This work is licensed under the GNU General Public License version 3.

from scipy import linalg
import numpy as np
np.set_printoptions (suppress = True, linewidth = 256)

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

# accepts a monzo, returns the ratio in [num, den] form
def monzo2ratio (monzo):
    ratio = [1, 1]
    for i in range (len (monzo)):
        if monzo[i] > 0:
            ratio[0] *= PRIME_LIST[i]**monzo[i]
        elif monzo[i] < 0:
            ratio[1] *= PRIME_LIST[i]**(-monzo[i])
    return ratio

def find_temperamental_complexity (map, monzo, oe = True, show = True):
    weight_te = np.eye (map.shape[1])
    for i in range (map.shape[1]):
        try:
            weight_te[i][i] = 1/np.log2 (PRIME_LIST[i])
        except ZeroDivisionError:
            continue
    if oe: #octave equivalence
        map = map[1:]
    P = linalg.pinv (map @ weight_te @ weight_te @ map.T)
    t = map @ monzo #image
    complexity = np.sqrt (t.T @ P @ t)
    if show:
        ratio = monzo2ratio (monzo)
        print (f"{ratio[0]}/{ratio[1]}\t {complexity:.4f}")
    return [monzo, complexity]

def find_spectrum (map, monzo_list, oe = True):
    spectrum = []
    for i in range (monzo_list.shape[1]):
        spectrum.append (find_temperamental_complexity (map, monzo_list[:,i], show = False))
    spectrum.sort (key = lambda k: k[1])
    for i in range (len (spectrum)):
        ratio = monzo2ratio (spectrum[i][0])
        print (f"{ratio[0]}/{ratio[1]}\t{round (spectrum[i][1], 4)}")

# examples
MONZO9 = np.array ([[2, -1, 0, 0], [-2, 0, 1, 0], [1, 1, -1, 0], [3, 0, 0, -1], [-1, -1, 0, 1], [0, 0, -1, 1], [-3, 2, 0, 0], [1, -2, 1, 0], [0, 2, 0, -1]]).T
MONZO11 = np.array ([[2, -1, 0, 0, 0], [-2, 0, 1, 0, 0], [1, 1, -1, 0, 0], [3, 0, 0, -1, 0], [-1, -1, 0, 1, 0], [0, 0, -1, 1, 0], [-3, 2, 0, 0, 0], [1, -2, 1, 0, 0], [0, 2, 0, -1, 0], \
[-3, 0, 0, 0, 1], [2, 1, 0, 0, -1], [0, -2, 0, 0, 1], [-1, 0, -1, 0, 1], [1, 0, 0, 1, -1]]).T
MONZO15 = np.array ([[2, -1, 0, 0, 0, 0], [-2, 0, 1, 0, 0, 0], [1, 1, -1, 0, 0, 0], [3, 0, 0, -1, 0, 0], [-1, -1, 0, 1, 0, 0], [0, 0, -1, 1, 0, 0], [-3, 2, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 2, 0, -1, 0, 0], \
[-3, 0, 0, 0, 1, 0], [2, 1, 0, 0, -1, 0], [0, -2, 0, 0, 1, 0], [-1, 0, -1, 0, 1, 0], [1, 0, 0, 1, -1, 0], \
[-3, 0, 0, 0, 0, 1], [-2, -1, 0, 0, 0, 1], [1, 2, 0, 0, 0, -1], [-1, 0, -1, 0, 0, 1], [1, 0, 0, 1, 0, -1], \
[4, -1, -1, 0, 0, 0], [-1, 1, 1, -1, 0, 0], [0, 1, 1, 0, -1, 0], [0, 1, 1, 0, 0, -1]]).T

map_history13 = np.array ([[1, 2, 0, 0, 1, 2], [0, 6, 0, -7, -2, 9], [0, 0, 1, 1, 1, 1]])
find_spectrum (map_history13, MONZO15)
