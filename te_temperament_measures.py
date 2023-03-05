# © 2020-2023 Flora Canou | Version 0.24.0
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class Temperament:
    def __init__ (self, val_list, subgroup = None, saturate = True, normalize = True):
        val_list, subgroup = te.get_subgroup (val_list, subgroup, axis = te.ROW)
        self.subgroup = subgroup
        self.jip = np.log2 (self.subgroup)*te.SCALAR
        self.map = te.canonicalize (np.rint (val_list).astype (int), saturate, normalize)

    class __Norm: 
        def __init__ (self, wtype = "tenney", wamount = 1, skew = 0, order = 2):
            self.wtype = wtype
            self.wamount = wamount
            self.skew = skew
            self.order = order

    def weightskewed (self, main, wtype = "tenney", wamount = 1, skew = 0, order = 2):
        return te.weightskewed (main, self.subgroup, wtype, wamount, skew, order)

    # checks availability of the symbolic solver
    def __check_sym (self, order):
        if order == 2:
            try:
                global te_sym
                import te_symbolic as te_sym
            except ImportError:
                warnings.warn ("Module te_symbolic.py not found. Using main optimizer. ")
                return False
        else:
            warnings.warn ("Condition for symbolic solution not met. Using main optimizer. ")
            return False
        return True

    # interprets the enforce specification as a monzo
    def __get_enforce_vec (self, enforce_index, wtype, wamount, skew, order, optimizer):
        if optimizer == "main":
            if enforce_index == 0:
                weightskew = self.weightskewed (np.eye (len (self.subgroup)), wtype, wamount, skew, order)
                return weightskew @ np.ones (weightskew.shape[1])
            else:
                return np.array ([1 if i == enforce_index - 1 else 0 for i, _ in enumerate (self.subgroup)])
        elif optimizer == "sym":
            if enforce_index == 0:
                weightskew = te_sym.get_weight_sym (self.subgroup, wtype, wamount) @ te_sym.get_skew_sym (self.subgroup, skew)
                return weightskew @ Matrix.ones (weightskew.shape[1], 1)
            else:
                return Matrix ([1 if i == enforce_index - 1 else 0 for i, _ in enumerate (self.subgroup)])

    # this mean rejects the extra dimension from the denominator
    # such that when skew = 0, introducing the extra dimension doesn't change the result
    def __mean (self, main):
        return np.sum (main)/len (self.subgroup)

    def __power_mean_norm (self, main, order):
        if order == np.inf:
            return np.max (main)
        else:
            return np.power (self.__mean (np.power (np.abs (main), order)), np.reciprocal (float (order)))

    def __show_header (self, norm = None, enforce_text = None, ntype = None):
        print (f"\nSubgroup: {'.'.join (map (str, self.subgroup))}",
            f"Mapping: \n{self.map}", sep = "\n")

        if norm: 
            weight_text = norm.wtype
            if norm.wamount != 1:
                weight_text += f"[{norm.wamount}]"

            skew_text = ""
            if norm.skew != 0:
                skew_text += "-weil"
                if norm.skew != 1:
                    skew_text += f"[{norm.skew}]"

            order_dict = {1: "-chebyshevian", 2: "-euclidean", np.inf: "-manhattan"}
            try:
                order_text = order_dict[norm.order]
            except KeyError:
                order_text = f"-L{norm.order}"

            print (f"Norm: {weight_text}{skew_text}{order_text}")
        if enforce_text:
            print (f"Enforcement: {enforce_text}")
        if ntype:
            print (f"Normalizer: {ntype}")
        return

    def tune (self, optimizer = "main", wtype = "tenney", wamount = 1, skew = 0, order = 2,
            enforce = "", cons_monzo_list = None, des_monzo = None): #in cents
        # checks optimizer availability
        if optimizer == "sym" and not self.__check_sym (order):
            return self.tune (optimizer = "main", wtype = wtype, wamount = wamount, skew = skew, order = order,
                enforce = enforce, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)

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
                        cons_monzo_list.append (self.__get_enforce_vec (int (entry[1:]), wtype, wamount, skew, order, optimizer))
                        cons_text.append ("WXj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}")
                    else:
                        des_monzo.append (self.__get_enforce_vec (int (entry[1:]), wtype, wamount, skew, order, optimizer))
                        des_text.append ("WXj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}")
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

        # shows the header
        self.__show_header (norm = self.__Norm (wtype, wamount, skew, order), enforce_text = enforce_text)

        # optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tuning_map, mistuning_map = te_opt.optimizer_main (self.map, subgroup = self.subgroup,
                wtype = wtype, wamount = wamount, skew = skew, order = order,
                cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)
        elif optimizer == "sym":
            gen, tuning_map, mistuning_map = te_sym.symbolic (self.map, subgroup = self.subgroup,
                wtype = wtype, wamount = wamount, skew = skew,
                cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)

        # error and bias
        tuning_map_wx = self.weightskewed (tuning_map, wtype, wamount, skew, order)
        mistuning_map_wx = self.weightskewed (mistuning_map, wtype, wamount, skew, order)
        error = self.__power_mean_norm (mistuning_map_wx, order)
        bias = np.mean (mistuning_map_wx)
        # print (mistuning_map_wx) #debug
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")
        return gen, tuning_map, mistuning_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, wtype = "tenney", wamount = 1, skew = 0, show = True):
        combination_list = itertools.combinations (range (self.map.shape[1]), self.map.shape[0])
        wedgie = np.array ([linalg.det (self.weightskewed (self.map, wtype, wamount, skew)[:, entry]) for entry in combination_list])
        wedgie *= np.copysign (1, wedgie[0])
        if show:
            self.__show_header ()
            print (f"Wedgie: {wedgie}", sep = "\n")
        return wedgie

    def complexity (self, ntype = "breed", wtype = "tenney", wamount = 1, skew = 0):
        # standard L2 complexity
        complexity = np.sqrt (linalg.det (
            self.weightskewed (self.map, wtype, wamount, skew)
            @ self.weightskewed (self.map, wtype, wamount, skew).T))
        # complexity = linalg.norm (self.wedgie (wtype, wamount, skew)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.map.shape[1]**self.map.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (tuple (itertools.combinations (range (self.map.shape[1]), self.map.shape[0]))))
        elif ntype == "l2":
            pass
        else:
            warnings.warn ("average type not supported, using default (\"breed\")")
            return self.complexity (ntype = "breed", wtype = wtype, wamount = wamount, skew = skew)
        return complexity

    def error (self, ntype = "breed", wtype = "tenney", wamount = 1, skew = 0): #in cents
        # standard L2 error
        error = linalg.norm (
            self.weightskewed (self.jip, wtype, wamount, skew)
            @ linalg.pinv (self.weightskewed (self.map, wtype, wamount, skew))
            @ self.weightskewed (self.map, wtype, wamount, skew)
            - self.weightskewed (self.jip, wtype, wamount, skew))
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
            warnings.warn ("average type not supported, using default (\"breed\")")
            return self.error (ntype = "breed", wtype = wtype, wamount = wamount, skew = skew)
        return error

    def badness (self, ntype = "breed", wtype = "tenney", wamount = 1, skew = 0): #in octaves
        return (self.error (ntype, wtype, wamount, skew)
            * self.complexity (ntype, wtype, wamount, skew)
            / te.SCALAR)

    def badness_logflat (self, ntype = "breed", wtype = "tenney", wamount = 1, skew = 0): #in octaves
        try:
            return (self.error (ntype, wtype, wamount, skew)
                * self.complexity (ntype, wtype, wamount, skew)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1)
                / te.SCALAR)
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", wtype = "tenney", wamount = 1, skew = 0, badness_scale = 1000):
        # shows the header
        self.__show_header (norm = self.__Norm (wtype, wamount, skew, 2), ntype = ntype)

        # shows the temperament measures
        error = self.error (ntype, wtype, wamount, skew)
        complexity = self.complexity (ntype, wtype, wamount, skew)
        badness = self.badness (ntype, wtype, wamount, skew) * badness_scale
        badness_logflat = self.badness_logflat (ntype, wtype, wamount, skew) * badness_scale
        print (f"Complexity: {complexity:.6f}",
            f"Error: {error:.6f} (¢)",
            f"Badness (simple): {badness:.6f} ({badness_scale}oct)",
            f"Badness (logflat): {badness_logflat:.6f} ({badness_scale}oct)", sep = "\n")

    def comma_basis (self, show = True):
        comma_basis_frac = Matrix (self.map).nullspace ()
        comma_basis = np.column_stack ([te.matrix2array (entry) for entry in comma_basis_frac])
        if show:
            self.__show_header ()
            print ("Comma basis: ")
            te.show_monzo_list (comma_basis_frac, self.subgroup)
        return comma_basis
