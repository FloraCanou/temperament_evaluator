# © 2020-2022 Flora Canou | Version 0.21.0
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class Temperament:
    def __init__ (self, map, subgroup = None, normalize = True):
        map, subgroup = te.get_subgroup (np.array (map), subgroup, axis = te.ROW)
        self.subgroup = subgroup
        self.jip = np.log2 (self.subgroup)*te.SCALAR
        self.map = te.normalize (np.rint (map).astype (np.int), axis = te.ROW) if normalize else map

    def weightskewed (self, main, wtype = "tenney", skew = 0, order = 2):
        return te.weightskewed (main, self.subgroup, wtype, skew, order)

    # checks availability of the symbolic solver
    def __check_sym (self, order):
        if order == 2:
            try:
                import te_symbolic as te_sym
                global te_sym
            except ImportError:
                warnings.warn ("Module te_symbolic.py not found. Using main optimizer. ")
                return False
        else:
            warnings.warn ("Condition for symbolic solution not met. Using main optimizer. ")
            return False
        return True

    # interprets the enforce specification as a monzo
    def __get_enforce_vec (self, enforce_index, wtype, optimizer):
        if optimizer == "main":
            if enforce_index == 0:
                return self.weightskewed (np.ones (len (self.subgroup)), wtype)
            else:
                return np.array ([1 if i == enforce_index - 1 else 0 for i, _ in enumerate (self.subgroup)])
        elif optimizer == "sym":
            if enforce_index == 0:
                return te_sym.get_weight_sym (self.subgroup, wtype) @ Matrix.ones (len (self.subgroup), 1)
            else:
                return Matrix ([1 if i == enforce_index - 1 else 0 for i, _ in enumerate (self.subgroup)])

    # this mean rejects the extra dimension from the denominator
    # such that when skew = 0, introducing the extra dimension doesn't change the same result
    def __mean (self, main):
        return np.sum (main)/len (self.subgroup)

    def __power_mean_norm (self, main, order):
        return np.power (self.__mean (np.power (np.abs (main), order)), np.reciprocal (float (order)))

    def tune (self, optimizer = "main", wtype = "tenney", skew = 0, order = 2,
            enforce = "", cons_monzo_list = None, des_monzo = None): #in cents
        # checks optimizer availability
        if optimizer == "sym" and not self.__check_sym (order):
            return self.tune (optimizer = "main", wtype = wtype, order = order,
                enforce = enforce, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo, skew = skew)

        # gets the enforcements
        if enforce == "c": #default to octave-constrained
            enforce = "c1"
        elif enforce == "d": #default to octave-destretched
            enforce = "d1"
        if cons_monzo_list is None and des_monzo is None:
            if enforce_spec_list := re.findall ("[cd]\d+", str (enforce)): #separates the enforcements
                cons_monzo_list, des_monzo, cons_text, des_text = ([] for _ in range (4))
                for entry in enforce_spec_list:
                    if entry[0] == "c":
                        cons_monzo_list.append (self.__get_enforce_vec (int (entry[1:]), wtype, optimizer))
                        cons_text.append ("Wj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}")
                    else:
                        des_monzo.append (self.__get_enforce_vec (int (entry[1:]), wtype, optimizer))
                        des_text.append ("Wj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}")
                if optimizer == "main":
                    cons_monzo_list = np.column_stack (cons_monzo_list) if cons_monzo_list else None
                    des_monzo = np.column_stack (des_monzo) if des_monzo else None
                elif optimizer == "sym":
                    cons_monzo_list = Matrix (BlockMatrix (cons_monzo_list)) if cons_monzo_list else None
                    des_monzo = Matrix (BlockMatrix (des_monzo)) if des_monzo else None
                cons_text = ".".join (cons_text) + "-constrained" if cons_text else ""
                des_text = ".".join (des_text) + "-destretched" if des_text else ""
                enforce_text = " ".join ([cons_text, des_text])
            else:
                enforce_text = "none"
        else:
            enforce_text = "custom"

        # header
        order_dict = {1: "-chebyshevian", 2: "-euclidean", np.inf: "-minkowskian"}
        try:
            order_text = order_dict[order]
        except KeyError:
            order_text = f"-L{order}"
        skew_dict = {0: "", 1: "-weil"}
        try:
            skew_text = skew_dict[skew]
        except KeyError:
            skew_text = f"-{skew}"
        print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
            f"Mapping: \n{self.map}",
            f"Norm: {wtype}{skew_text}{order_text}. Enforcement: {enforce_text}", sep = "\n")

        # optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tuning_map, mistuning_map = te_opt.optimizer_main (self.map, subgroup = self.subgroup,
                wtype = wtype, skew = skew, order = order, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)
        elif optimizer == "sym":
            gen, tuning_map, mistuning_map = te_sym.symbolic (self.map, subgroup = self.subgroup,
                wtype = wtype, skew = skew, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)

        # error and bias
        tuning_map_wx = self.weightskewed (tuning_map, wtype, skew, order)
        mistuning_map_wx = self.weightskewed (mistuning_map, wtype, skew, order)
        error = self.__power_mean_norm (mistuning_map_wx, order)
        bias = np.mean (mistuning_map_wx)
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")
        return gen, tuning_map, mistuning_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, wtype = "tenney", skew = 0, show = True):
        combination_list = itertools.combinations (range (self.map.shape[1]), self.map.shape[0])
        wedgie = np.array ([linalg.det (self.weightskewed (self.map, wtype, skew)[:, entry]) for entry in combination_list])
        wedgie *= np.copysign (1, wedgie[0])
        if show:
            print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
                f"Mapping: \n{self.map}",
                f"Wedgie: {wedgie}", sep = "\n")
        return wedgie

    def complexity (self, ntype = "breed", wtype = "tenney", skew = 0):
        # standard L2 complexity
        complexity = np.sqrt (linalg.det (self.weightskewed (self.map, wtype, skew) @ self.weightskewed (self.map, wtype, skew).T))
        # complexity = linalg.norm (self.wedgie (wtype = wtype)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.map.shape[1]**self.map.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (tuple (itertools.combinations (range (self.map.shape[1]), self.map.shape[0]))))
        elif ntype == "l2":
            pass
        else:
            warnings.warn ("norm type not supported, using default (\"breed\")")
            return self.complexity (ntype = "breed", wtype = wtype, skew = skew)
        return complexity

    def error (self, ntype = "breed", wtype = "tenney", skew = 0): #in cents
        # standard L2 error
        error = linalg.norm (
            self.weightskewed (self.jip, wtype, skew)
            @ linalg.pinv (self.weightskewed (self.map, wtype, skew))
            @ self.weightskewed (self.map, wtype, skew)
            - self.weightskewed (self.jip, wtype, skew))
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
            return self.error (ntype = "breed", wtype = wtype, skew = skew)
        return error

    def badness (self, ntype = "breed", wtype = "tenney", skew = 0): #in octaves
        return (self.error (ntype, wtype, skew)
            * self.complexity (ntype, wtype, skew)
            / te.SCALAR)

    def badness_logflat (self, ntype = "breed", wtype = "tenney", skew = 0): #in octaves
        try:
            return (self.error (ntype, wtype, skew)
                * self.complexity (ntype, wtype, skew)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1)
                / te.SCALAR)
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", wtype = "tenney", skew = 0, badness_scale = 1000):
        print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
            f"Mapping: \n{self.map}",
            f"Norm: {ntype}. Weighter: {wtype}", sep = "\n")

        error = self.error (ntype, wtype, skew)
        complexity = self.complexity (ntype, wtype, skew)
        badness = self.badness (ntype, wtype, skew) * badness_scale
        badness_logflat = self.badness_logflat (ntype, wtype, skew) * badness_scale
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
