import te_temperament_measures as tm
import et_sequence_error as et
import numpy as np

tm.et_construct (17, [2, 3, 5, 7, 11, 13], alt_val = [0, 0, 1, 0, 0, 0]).show_all (badness_scale = 100) # 17edo in 17c val

tm.Temperament ([[1, 0, 2, -1], [0, 5, 1, 12]]).show_all (type = "rmsgene") # septimal magic

# Monzos should be entered as column vectors
et.et_sequence_error (np.transpose ([[-4, 4, -1, 0], [-5, 2, 2, -1]]), cond = "error", search_range = 300) # septimal meantone
