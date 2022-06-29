# © 2020-2022 Flora Canou | Version 0.20.0
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix
from sympy import gcd
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class Temperament:
    def __init__ (self, map, subgroup = None, normalize = True):
        map, subgroup = te.get_subgroup (np.array (map), subgroup, axis = te.ROW)
        self.subgroup = subgroup
        self.jip = np.log2 (self.subgroup)*te.SCALAR
        self.map = te.normalize (np.rint (map).astype (np.int), axis = te.ROW) if normalize else map

    def weighted (self, main, wtype, *, k = 0.5):
        return main @ te.get_weight (self.subgroup, wtype = wtype, k = k)

    def __check_sym (self, wtype, order, des_monzo):
        return wtype in te.SYM_WEIGHT_LIST and order == 2 and des_monzo is None

    def __get_enforce_vec (self, enforce_index, wtype = "tenney"):
        if enforce_index == 0:
            if wtype in te.SYM_WEIGHT_LIST: #branch for symbolic calculations
                enforce_vec = self.weighted (np.lcm.reduce (self.subgroup)*np.ones (len (self.subgroup)), wtype = wtype).astype (int)
            else:
                enforce_vec = self.weighted (np.ones (len (self.subgroup)), wtype = wtype)
        else:
            enforce_vec = np.zeros (len (self.subgroup), dtype = int)
            enforce_vec[enforce_index - 1] = 1
        return enforce_vec

    def tune (self, optimizer = "main", wtype = "tenney", order = 2,
            enforce = "", cons_monzo_list = None, des_monzo = None, *, k = 0.5): #in cents
        # enforcements
        if enforce == "c": #default to octave-constrained
            enforce = "c1"
        elif enforce == "d": #default to octave-destretched
            enforce = "d1"
        elif enforce == "xoc":
            enforce = "c0"
            warnings.warn ("\"xoc\" has been deprecated. Use \"c0\" instead.", FutureWarning)
        if cons_monzo_list is None and des_monzo is None:
            cons_text = des_text = ""
            if cons_spec_list := re.findall ("c\d+", str (enforce)): #constrained
                cons_monzo_list = np.column_stack ([self.__get_enforce_vec (int (entry[1:]), wtype) for entry in cons_spec_list])
                cons_text = ".".join (["Wj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}" for entry in cons_spec_list]) + "-constrained "
            if des_spec_list := re.findall ("d\d+", str (enforce)): #destretched
                des_monzo = np.column_stack ([self.__get_enforce_vec (int (entry[1:]), wtype) for entry in des_spec_list])
                des_text = ".".join (["Wj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}" for entry in des_spec_list]) + "-destretched "
            enforce_text = cons_text + des_text if cons_text + des_text else "none"
        else:
            enforce_text = "custom"

        # headers
        order_dict = {1: "chebyshevian", 2: "euclidean", np.inf: "minkowskian"}
        try:
            order_text = order_dict[order]
        except KeyError:
            order_text = f"L{order}"
        print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
            f"Mapping: \n{self.map}",
            f"Weight/norm: {wtype}-{order_text}. Enforcement: {enforce_text}", sep = "\n")

        if optimizer == "sym":
            if self.__check_sym (wtype, order, des_monzo):
                try:
                    import te_symbolic as te_sym
                    gen, tuning_map, mistuning_map = te_sym.symbolic (self.map, subgroup = self.subgroup,
                        wtype = wtype, cons_monzo_list = cons_monzo_list)
                except ImportError:
                    optimizer = "main"
                    warnings.warn ("Module te_symbolic.py not found. Using main optimizer. ")
            else:
                optimizer = "main"
                print ("Condition for symbolic solution not met. Using main optimizer. ")
        if optimizer == "main":
            if self.__check_sym (wtype, order, des_monzo):
                print ("Tip: symbolic solution is available. Set optimizer to \"sym\" to use. ")
            import te_optimizer as te_opt
            gen, tuning_map, mistuning_map = te_opt.optimizer_main (self.map, subgroup = self.subgroup,
                wtype = wtype, order = order, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo, k = k)

        # error and bias
        tuning_map_w = self.weighted (tuning_map, wtype, k = k)
        mistuning_map_w = self.weighted (mistuning_map, wtype, k = k)
        error = linalg.norm (mistuning_map_w, ord = order) / np.sqrt (self.map.shape[1])
        bias = np.mean (mistuning_map_w)

        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")
        return gen, tuning_map, mistuning_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, wtype = "tenney", *, k = 0.5):
        combination_list = itertools.combinations (range (self.map.shape[1]), self.map.shape[0])
        wedgie = np.array ([linalg.det (self.weighted (self.map, wtype, k = k)[:, entry]) for entry in combination_list])
        return wedgie if wedgie[0] >= 0 else -wedgie

    def complexity (self, ntype = "breed", wtype = "tenney", *, k = 0.5):
        # standard L2 complexity
        complexity = np.sqrt (linalg.det (self.weighted (self.map, wtype, k = k) @ self.weighted (self.map, wtype, k = k).T))
        # complexity = linalg.norm (self.wedgie (wtype = wtype)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.map.shape[1]**self.map.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (self.wedgie ()))
        elif ntype == "l2":
            pass
        else:
            warnings.warn ("norm type not supported, using default (\"breed\")")
            return self.complexity (ntype = "breed", wtype = wtype, k = k)
        return complexity

    def error (self, ntype = "breed", wtype = "tenney", *, k = 0.5): #in cents
        # standard L2 error
        error = linalg.norm (
            self.weighted (self.jip, wtype, k = k)
            @ linalg.pinv (self.weighted (self.map, wtype, k = k))
            @ self.weighted (self.map, wtype, k = k)
            - (self.weighted (self.jip, wtype, k = k)))
        if ntype == "breed": #Graham Breed's RMS (default)
            error *= 1/np.sqrt (self.map.shape[1])
        elif ntype == "smith": #Gene Ward Smith's RMS
            try:
                error *= np.sqrt ((self.map.shape[0] + 1) / (self.map.shape[1] - self.map.shape[0]))
            except ZeroDivisionError:
                error = np.nan
        elif ntype == "l2":
            pass
        else:
            warnings.warn ("norm type not supported, using default (\"breed\")")
            return self.error (ntype = "breed", wtype = wtype, k = k)
        return error

    def badness (self, ntype = "breed", wtype = "tenney", *, k = 0.5): #in octaves
        return (self.error (ntype = ntype, wtype = wtype, k = k)
            * self.complexity (ntype = ntype, wtype = wtype, k = k)
            / te.SCALAR)

    def badness_logflat (self, ntype = "breed", wtype = "tenney", *, k = 0.5): #in octaves
        try:
            return (self.error (ntype = ntype, wtype = wtype, k = k)
                * self.complexity (ntype = ntype, wtype = wtype, k = k)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1)
                / te.SCALAR)
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", wtype = "tenney", badness_scale = 100, *, k = 0.5):
        print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
            f"Mapping: \n{self.map}",
            f"Norm: {ntype}. Weighter: {wtype}", sep = "\n")

        error = self.error (ntype = ntype, wtype = wtype, k = k)
        complexity = self.complexity (ntype = ntype, wtype = wtype, k = k)
        badness = self.badness (ntype = ntype, wtype = wtype, k = k) * badness_scale
        badness_logflat = self.badness_logflat (ntype = ntype, wtype = wtype, k = k) * badness_scale
        print (f"Complexity: {complexity:.6f}",
            f"Error: {error:.6f} (¢)",
            f"Badness (simple): {badness:.6f} ({badness_scale}oct)",
            f"Badness (logflat): {badness_logflat:.6f} ({badness_scale}oct)", sep = "\n")

    def comma_basis (self, show = True):
        comma_basis_frac = Matrix (self.map).nullspace ()
        comma_basis = np.transpose ([np.squeeze (entry/gcd (tuple (entry))) for entry in comma_basis_frac])
        if show:
            print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
                f"Mapping: \n{self.map}",
                "Comma basis: ", sep = "\n")
            for i in range (comma_basis.shape[1]):
                monzo = comma_basis[:, i]
                ratio = te.monzo2ratio (monzo, self.subgroup)
                monzo_str = "[" + " ".join (map (str, np.trim_zeros (monzo, trim = "b"))) + ">"
                if ratio[0] < 10e7:
                    print (monzo_str, f"({ratio[0]}/{ratio[1]})")
                else:
                    print (monzo_str)
        return comma_basis
