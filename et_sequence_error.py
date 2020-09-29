# Copyright 2020 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import te_temperament_measures as tm
import numpy as np

# Find et sequence from comma list. Can be used to find optimal patent vals
def et_sequence_error (monzo_list, subgroup, cond = "error", threshold = 10, search_range = 1200):
    et_list = []
    ed_ratio = 2
    for n in range (search_range):
        et_list.append (tm.et_construct (n, 2, subgroup))
        if n != 0 and any ((et_list[n].map @ monzo_list)[0]) == False:
            if cond == "error":
                if et_list[n].error () <= threshold:
                    threshold = et_list[n].error ()
                    et_list[n].show_temperament_measures ()
            elif cond == "simple_badness":
                if et_list[n].simple_badness () <= threshold:
                    threshold = et_list[n].simple_badness ()
                    et_list[n].show_temperament_measures ()
            elif cond == "all":
                et_list[n].show_temperament_measures ()
