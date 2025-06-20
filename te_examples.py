import numpy as np
import te_common as te
import te_temperament_measures as te_tm
import te_equal as te_et
import te_lattice as te_lat

# Set precision here
np.set_printoptions (precision = 3)

# Important: a single monzo should be entered as a vector; a monzo list should be entered as composed by column vectors

# Norm
# parameters: 
#   wtype: weight method. "tenney", "equilateral" and "wilson"/"benedetti"
#   wamount: weight scaling factor
#   skew: skew of the space
#   order: optimization order

# Temperament
# parameters:
#   subgroup: specifies a custom ji subgroup
# methods:
#   tune: gives the tuning
#     parameters:
#       optimizer: "main", for the main solver, or "sym", for the symbolic solver
#       norm: norm profile of the tuning space. see above
#       inharmonic: for subgroup temps, treats the basis as if they were primes
#       constraint: constrains this subgroup to pure
#       destretch: destretches this ratio to pure
#   temperament_measures: gives the temperament measures
#     parameters:
#       ntype: averaging normalizer. "breed", "smith", "sintel", or "none"
#       norm: norm profile for the tuning space. see above
#       error_scale: scales the error
#       badness_scale: scales the badness
#   comma_basis: gives the comma basis

temp = te_tm.Temperament ([
    [1, 0, 2, -1], 
    [0, 5, 1, 12]
    ]) # septimal magic
temp.tune (norm = te.Norm (skew = 1), constraint = te.Subgroup ("2")) # cwe a.k.a. ke tuning
temp.temperament_measures (ntype = "sintel", badness_scale = 1) # te temperament measures
temp.wedgie ()
temp.comma_basis ()

# et_construct
# parameters:
#   subgroup: custom ji subgroup

temp = te_et.et_construct (["14c", "17c"], te.Subgroup ([2, 3, 5, 7])) # squares
temp.temperament_measures (ntype = "sintel", badness_scale = 1)

temp = te_et.et_construct (["b4", "b13"], te.Subgroup ([3, 5, 7])) # bps
temp.temperament_measures (ntype = "sintel", badness_scale = 1)

# comma_construct
# parameters:
#   subgroup: custom ji subgroup

temp = te_et.comma_construct (te.Subgroup ([
    "126/125", 
    "245/243"
    ]).basis_matrix) # septimal sensi
temp.temperament_measures (ntype = "sintel", badness_scale = 1)

# et_sequence
# parameters:
#   subgroup: custom ji subgroup
#   ntype: averaging normalizer. "breed", "smith", "sintel", or "none"
#   norm: norm profile for the tuning space. see above
#   inharmonic: for subgroup temps, treats the basis as if they were primes
#   cond: "error", "badness", or "logflat badness"
#   threshold: temperaments failing this will not be shown
#   prog: if true, threshold will be updated on iteration
#   pv: if true, only patent vals will be considered
#   search_range: upper bound where to stop searching

temp = te_tm.Temperament ([
    [1, 0, -4, -13], 
    [0, 1, 4, 10]
    ]) # septimal meantone
te_et.et_sequence (temp.comma_basis (show = False), cond = "error", search_range = 300)

# find_complexity_spectrum
# parameters:
#   norm: norm profile for the tuning space. see above
#   oe: octave equivalence
temp = te_lat.TemperamentLattice ([
    [1, 2, 0, 0, 1, 2], 
    [0, 6, 0, -7, -2, 9], 
    [0, 0, 1, 1, 1, 1]
    ]) # tridecimal history
temp.find_complexity_spectrum (te_lat.odd_limit_monzos_gen (15), oe = True)
