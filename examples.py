import numpy as np
from lib.te_common import *
from lib.te_temperament_measures import *
from lib.te_equal import *
from lib.te_lattice import *

# set precision here
np.set_printoptions (precision = 3)

# basic usages

## to construct a temperament
## we're showing septimal magic here
temp = Temperament (
    [[1, 0, 2, -1], 
    [0, 5, 1, 12]]) 

## to tune the temperament
### cte tuning
temp.tune (constraint = Subgroup ([2])) 

### cwe a.k.a. ke tuning
temp.tune (norm = Norm (skew = 1), constraint = Subgroup ([2])) 

### pote tuning
temp.tune (destretch = Ratio (2, 1)) 

## to find the te temperament measures
temp.temperament_measures (ntype = "sintel") 

## to find the wedgie of the temperament
temp.wedgie ()

## to find a comma basis of the temperament
temp.comma_basis ()

## to find the optimal GPV sequence of the temperament
et_sequence (
    temp.comma_basis (show = False), cond = "error", search_range = 300)

# alternative ways to construct temperaments

## to construct a temperament from equal temperaments
### squares
temp = et_construct (["14c", "17c"], Subgroup ([2, 3, 5, 7]))
temp.temperament_measures (ntype = "sintel")

### bps
temp = et_construct (["b4", "b13"], Subgroup ([3, 5, 7]))
temp.temperament_measures (ntype = "sintel")

## to construct a temperament from a comma basis
### septimal sensi
temp = comma_construct (
    Subgroup (["126/125", "245/243"]).basis_matrix)
temp.temperament_measures (ntype = "sintel")

# lattice-related functions

## this is mainly used to find the octave-equivalent interval temperamental complexity spectrum
## to do this, we need to construct the temperament with te_lat.TemperamentLattice
## here we're demonstrating with tridecimal history
temp = TemperamentLattice (
    [[1, 2, 0, 0, 1, 2], 
    [0, 6, 0, -7, -2, 9], 
    [0, 0, 1, 1, 1, 1]]
)
temp.temperamental_complexity_spectrum (
    diamond_monzos_gen (15, eq = 2, comp = False), oe = True)
