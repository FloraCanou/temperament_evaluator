# © 2020-2023 Flora Canou | Version 0.25.1
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class Temperament:
    def __init__ (self, vals, subgroup = None, saturate = True, normalize = True):
        vals, subgroup = te.get_subgroup (vals, subgroup, axis = te.ROW)
        self.subgroup = subgroup
        self.jip = np.log2 (self.subgroup)*te.SCALAR
        self.map = te.canonicalize (np.rint (vals).astype (int), saturate, normalize)

    def weightskewed (self, main, norm):
        return te.weightskewed (main, self.subgroup, norm)

    # checks availability of the symbolic solver
    def __check_sym (self, order):
        if order == 2:
            try:
                global te_sym
                import te_symbolic as te_sym
            except ImportError:
                warnings.warn ("Module te_symbolic.py not found. Using main optimizer. ")
                return False
            return True
        else:
            warnings.warn ("Condition for symbolic solution not met. Using main optimizer. ")
            return False

    # interprets the enforce specification as a monzo
    def __get_enforce_vec (self, enforce_index, norm, optimizer):
        if optimizer == "main":
            if enforce_index == 0:
                weightskew = self.weightskewed (np.eye (len (self.subgroup)), norm)
                return weightskew @ np.ones (weightskew.shape[1])
            else:
                return np.array ([1 if i == enforce_index - 1 else 0 for i, _ in enumerate (self.subgroup)])
        elif optimizer == "sym":
            if enforce_index == 0:
                weightskew = (
                    te_sym.__get_weight_sym (self.subgroup, norm.wtype, norm.wamount)
                    @ te_sym.__get_skew_sym (self.subgroup, norm.skew, norm.order))
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

    def tune (self, optimizer = "main", norm = te.Norm (), 
            enforce = "", cons_monzo_list = None, des_monzo = None, 
            *, wtype = None, wamount = None, skew = None, order = None): #in cents

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order
        
        # checks optimizer availability
        if optimizer == "sym" and not self.__check_sym (norm.order):
            return self.tune (optimizer = "main", norm = norm,
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
                        cons_monzo_list.append (self.__get_enforce_vec (int (entry[1:]), norm, optimizer))
                        cons_text.append ("WXj" if entry[1:] == "0" else f"{self.subgroup[int (entry[1:]) - 1]}")
                    else:
                        des_monzo.append (self.__get_enforce_vec (int (entry[1:]), norm, optimizer))
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
        self.__show_header (norm = norm, enforce_text = enforce_text)

        # optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tuning_map, mistuning_map = te_opt.optimizer_main (self.map, subgroup = self.subgroup,
                norm = norm, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)
        elif optimizer == "sym":
            gen, tuning_map, mistuning_map = te_sym.symbolic (self.map, subgroup = self.subgroup,
                norm = norm, cons_monzo_list = cons_monzo_list, des_monzo = des_monzo)

        # error and bias
        tuning_map_wx = self.weightskewed (tuning_map, norm)
        mistuning_map_wx = self.weightskewed (mistuning_map, norm)
        error = self.__power_mean_norm (mistuning_map_wx, norm.order)
        bias = np.mean (mistuning_map_wx)
        # print (mistuning_map_wx) #debug
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")
        return gen, tuning_map, mistuning_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, norm = te.Norm (), show = True, 
            *, wtype = None, wamount = None, skew = None, order = None):

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        combination_list = itertools.combinations (range (self.map.shape[1]), self.map.shape[0])
        wedgie = np.array ([linalg.det (self.weightskewed (self.map, norm)[:, entry]) for entry in combination_list])
        wedgie *= np.copysign (1, wedgie[0])
        if show:
            self.__show_header ()
            print (f"Wedgie: {wedgie}", sep = "\n")
        return wedgie

    def complexity (self, ntype = "breed", norm = te.Norm (), 
            *, wtype = None, wamount = None, skew = None, order = None):
        if norm.order != 2:
            raise NotImplementedError ("order must be 2. ")
        
        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        # standard L2 complexity
        complexity = np.sqrt (linalg.det (
            self.weightskewed (self.map, norm)
            @ self.weightskewed (self.map, norm).T))
        # complexity = linalg.norm (self.wedgie (norm = norm, show = False)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.map.shape[1]**self.map.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (tuple (itertools.combinations (range (self.map.shape[1]), self.map.shape[0]))))
        elif ntype == "none":
            pass
        elif ntype == "l2":
            warnings.warn ("\"l2\" is deprecated. Use \"none\" instead. ")
        else:
            warnings.warn ("normalizer not supported, using default (\"breed\")")
            return self.complexity (ntype = "breed", norm = norm)
        return complexity

    def error (self, ntype = "breed", norm = te.Norm (), 
            *, wtype = None, wamount = None, skew = None, order = None): #in cents
        if norm.order != 2:
            raise NotImplementedError ("order must be 2. ")

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order
        
        # standard L2 error
        error = linalg.norm (
            self.weightskewed (self.jip, norm)
            @ linalg.pinv (self.weightskewed (self.map, norm))
            @ self.weightskewed (self.map, norm)
            - self.weightskewed (self.jip, norm))
        if ntype == "breed": #Graham Breed's RMS (default)
            error *= 1/np.sqrt (self.map.shape[1])
        elif ntype == "smith": #Gene Ward Smith's RMS
            try:
                error *= np.sqrt ((self.map.shape[0] + 1) / (self.map.shape[1] - self.map.shape[0]))
            except ZeroDivisionError:
                error = np.nan
        elif ntype == "none":
            pass
        elif ntype == "l2":
            warnings.warn ("\"l2\" is deprecated. Use \"none\" instead. ")
        else:
            warnings.warn ("normalizer not supported, using default (\"breed\")")
            return self.error (ntype = "breed", norm = norm)
        return error

    def badness (self, ntype = "breed", norm = te.Norm (), 
            *, wtype = None, wamount = None, skew = None, order = None): #in octaves

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        return (self.error (ntype, norm)
            * self.complexity (ntype, norm)
            / te.SCALAR)

    def badness_logflat (self, ntype = "breed", norm = te.Norm (), 
            *, wtype = None, wamount = None, skew = None, order = None): #in octaves

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        try:
            return (self.error (ntype, norm)
                * self.complexity (ntype, norm)**((self.map.shape[0])/(self.map.shape[1] - self.map.shape[0]) + 1)
                / te.SCALAR)
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", norm = te.Norm (), badness_scale = 1000, 
            *, wtype = None, wamount = None, skew = None, order = None):

        # DEPRECATION WARNING
        if any ((wtype, wamount, skew, order)): 
            warnings.warn ("\"wtype\", \"wamount\", \"skew\", and \"order\" are deprecated. Use the Norm class instead. ")
            if wtype: norm.wtype = wtype
            if wamount: norm.wamount = wamount
            if skew: norm.skew = skew
            if order: norm.order = order

        # shows the header
        self.__show_header (norm = norm, ntype = ntype)

        # shows the temperament measures
        error = self.error (ntype, norm)
        complexity = self.complexity (ntype, norm)
        badness = self.badness (ntype, norm) * badness_scale
        badness_logflat = self.badness_logflat (ntype, norm) * badness_scale
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
