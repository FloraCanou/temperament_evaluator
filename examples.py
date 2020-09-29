import te_temperament_measures as tm
from et_sequence_error import et_sequence_error
import numpy as np

# 17edo in 17c val
tm.et_construct (17, 2, [2, 3, 5, 7, 11, 13], alt_val = [0, 0, 1, 0, 0, 0]).show_all (badness_scale = 100)

# seven-limit magic
temper_magic7 = tm.Temperament ([[1, 0, 2, -1], [0, 5, 1, 12]], [2, 3, 5, 7])
temper_magic7.show_all (type = "rmsgene")

# septimal meantone
et_sequence_error (np.transpose ([[-4, 4, -1, 0], [-5, 2, 2, -1]]), [2, 3, 5, 7], cond = "error", search_range = 200)
