import te_temperament_measures as tm
import et_sequence_error as et
import numpy as np

# Note: monzos should be entered as column vectors

# Temperament
# parameters:
#   subgroup: specifies a custom ji subgroup
# methods:
#   analyse: gives the tuning
#     parameters:
#       type: "custom" , "te", "pote", "cte", "top", "potop", or "ctop"
#       order: specifies the order of the norm to be minimized
#       cons_monzo_list: constrains this list of monzos to pure
#       stretch_monzo: stretches this monzo to pure
#   temperament_measures: gives the te temperament measures
#     parameters:
#       type: "rmsgraham", "rmsgene", or "l2"
#       badness_scale: scales the badnesses, literally

tm.Temperament ([[1, 0, 2, -1], [0, 5, 1, 12]]).analyse (type = "cte") # septimal magic
tm.Temperament ([[1, 0, 2, -1], [0, 5, 1, 12]]).temperament_measures (type = "rmsgene")

# et_construct
# parameters:
#   alt_val: alter the val by this

et.et_construct (17, [2, 3, 5, 7, 11, 13], alt_val = [0, 0, 1, 0, 0, 0]).temperament_measures (badness_scale = 100) # 17edo in 17c val

# et_sequence_error
# parameters:
#   subgroup: specifies a custom ji subgroup
#   cond: "error" or "badness"
#   threshold: temperaments failing this will not be shown
#   progressive: if true (default), threshold will be updated
#   search_range: upper bound where to stop searching

et.et_sequence_error (np.transpose ([[-4, 4, -1, 0], [-5, 2, 2, -1]]), cond = "error", search_range = 300) # septimal meantone
