import numpy as np
import te_common as te
import te_temperament_measures as te_tm
import te_equal as te_et
import te_lattice as te_la

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
#       norm: norm profile of the tuning space. see above
#       constraint: constrains this subgroup to pure
#       destretch: destretches this ratio to pure
#   temperament_measures: gives the temperament measures
#     parameters:
#       ntype: averaging normalizer. "breed", "smith", or "l2"
#       norm: norm profile for the tuning space. see above
#       badness_scale: scales the badnesses, literally
#   comma_basis: gives the comma basis

temp = te_tm.Temperament ([
    [1, 0, 2, -1], 
    [0, 5, 1, 12]
    ]) # septimal magic
temp.tune (norm = te.Norm (skew = 1), constraint = te.Subgroup ("2")) # ctwe tuning
temp.temperament_measures (ntype = "smith", badness_scale = 1) # te temperament measures
temp.wedgie (norm = te.Norm (wtype = "equilateral"))
temp.comma_basis ()

# et_construct
# parameters:
#   subgroup: custom ji subgroup
#   alt_val: alters the val by this matrix

temp = te_et.et_construct (["14c", "17c"], te.Subgroup ([2, 3, 5, 7])) # squares
temp.temperament_measures (ntype = "smith", badness_scale = 1)

# comma_construct
# parameters:
#   subgroup: custom ji subgroup

temp = te_et.comma_construct (te.Subgroup ([
    "126/125", 
    "245/243"
    ]).basis_matrix) # septimal sensi
temp.temperament_measures (ntype = "smith", badness_scale = 1)

# et_sequence
# parameters:
#   subgroup: custom ji subgroup
#   ntype: averaging normalizer. "breed", "smith", or "none"
#   norm: norm profile for the tuning space. see above
#   cond: "error" or "badness"
#   threshold: temperaments failing this will not be shown
#   prog: if true, threshold will be updated
#   pv: if true, only patent vals will be considered
#   search_range: upper bound where to stop searching

temp = te_tm.Temperament ([
    [1, 0, -4, -13], 
    [0, 1, 4, 10]
    ]) # septimal meantone
te_et.et_sequence (temp.comma_basis (show = False), cond = "error", search_range = 300)

# find_spectrum
temp = te_la.TemperamentLattice ([
    [1, 2, 0, 0, 1, 2], 
    [0, 6, 0, -7, -2, 9], 
    [0, 0, 1, 1, 1, 1]
    ]) # history
temp.find_complexity_spectrum (te_la.odd_limit_monzo_list_gen (15))
