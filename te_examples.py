import numpy as np
import te_temperament_measures as te_tm
import te_equal as te_et
import te_lattice as te_la

# Note: a single monzo should be entered as a vector; a monzo list should be entered as composed by column vectors

# Temperament
# parameters:
#   subgroup: specifies a custom ji subgroup
# methods:
#   tune: gives the tuning
#     parameters:
#       wtype: specifies the weighter. "tenney", "frobenius", "inverse tenney", "benedetti" and "weil"
#       order: specifies the order of the norm to be minimized
#       enforce: specifies constraints and destretch targets. try "c", "d", "c1c2", "c0" and see readme for more examples
#       cons_monzo_list: constrains this list of monzos to pure
#       des_monzo: destretches this monzo to pure
#   temperament_measures: gives the temperament measures
#     parameters:
#       ntype: specifies the averaging method. "breed", "smith", or "l2"
#       wtype: specifies the weighter. "tenney", "frobenius", "inverse tenney", and "benedetti"
#       badness_scale: scales the badnesses, literally
#   comma_basis: gives the comma basis

A = te_tm.Temperament ([[1, 0, 2, -1], [0, 5, 1, 12]])
A.tune (skew = 1, enforce = "c") # septimal magic in ctwe tuning
A.temperament_measures (ntype = "smith")
A.wedgie (wtype = "frobenius")
A.comma_basis ()

# et_construct
# parameters:
#   subgroup: specifies a custom ji subgroup
#   alt_val: alters the val by this matrix

A = te_et.et_construct (["17c"], [2, 3, 5, 7, 11, 13])
A.temperament_measures (badness_scale = 100) # 17edo in 17c val

# et_sequence
# parameters:
#   subgroup: specifies a custom ji subgroup
#   ntype: specifies the averaging method. "breed", "smith", or "l2"
#   wtype: specifies the weighter. "tenney", "frobenius", "inverse tenney", and "benedetti"
#   cond: "error" or "badness"
#   threshold: temperaments failing this will not be shown
#   prog: if true, threshold will be updated
#   pv: if true, only patent vals will be considered
#   search_range: specifies the upper bound where to stop searching

A = te_tm.Temperament ([[1, 0, -4, -13], [0, 1, 4, 10]])
te_et.et_sequence (A.comma_basis (show = False), cond = "error", search_range = 300) # septimal meantone

# find_spectrum
A = te_la.TemperamentLattice ([[1, 2, 0, 0, 1, 2], [0, 6, 0, -7, -2, 9], [0, 0, 1, 1, 1, 1]])
A.find_spectrum (te_la.odd_limit_monzo_list_gen (15))
