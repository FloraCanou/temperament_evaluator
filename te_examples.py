import numpy as np
import te_common as te
import te_temperament_measures as te_tm
import te_equal as te_et
import te_lattice as te_lat

# set precision here
np.set_printoptions (precision = 3)

# NOTE: a single monzo should be entered as a vector; a monzo list should be entered as composed by column vectors

# Basic usages

## to construct a temperament
## we're showing septimal magic here
temp = te_tm.Temperament ([
    [1, 0, 2, -1], 
    [0, 5, 1, 12]
    ]) 

## to tune the temperament
### cte tuning
temp.tune (constraint = te.Subgroup ("2")) 

### cwe a.k.a. ke tuning
temp.tune (norm = te.Norm (skew = 1), constraint = te.Subgroup ("2")) 

### pote tuning
temp.tune (destretch = te.Ratio (2, 1)) 

## to find the te temperament measures
temp.temperament_measures (ntype = "sintel") 

## to find the wedgie of the temperament
temp.wedgie ()

## to find a comma basis of the temperament
temp.comma_basis ()

## to find the optimal GPV sequence of the temperament
te_et.et_sequence (temp.comma_basis (show = False), cond = "error", search_range = 300)

# Alternative ways to construct temperaments

## to construct a temperament from equal temperaments
### squares
temp = te_et.et_construct (["14c", "17c"], te.Subgroup ([2, 3, 5, 7]))
temp.temperament_measures (ntype = "sintel")

### bps
temp = te_et.et_construct (["b4", "b13"], te.Subgroup ([3, 5, 7]))
temp.temperament_measures (ntype = "sintel")

## to construct a temperament from a comma basis
### septimal sensi
temp = te_et.comma_construct (te.Subgroup ([
    "126/125", 
    "245/243"
    ]).basis_matrix)
temp.temperament_measures (ntype = "sintel")

# Lattice-related functions

## This is mainly used to find the octave-equivalent interval complexity spectrum. 
## To do this, we need to construct the temperament with `te_lat.TemperamentLattice`. 
## Here we're using tridecimal history. 
temp = te_lat.TemperamentLattice ([
    [1, 2, 0, 0, 1, 2], 
    [0, 6, 0, -7, -2, 9], 
    [0, 0, 1, 1, 1, 1]
    ]) 
temp.find_complexity_spectrum (te_lat.odd_limit_monzos_gen (15), oe = True)
