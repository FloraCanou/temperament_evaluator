# © 2020-2023 Flora Canou | Version 0.27.0
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class Temperament:
    def __init__ (self, vals, subgroup = None, saturate = True, normalize = True): #NOTE: "map" is a reserved word
        vals, subgroup = te.get_subgroup (vals, subgroup, axis = te.AXIS.ROW)
        self.subgroup = subgroup
        self.vals = te.canonicalize (np.rint (vals).astype (int), saturate, normalize)

    def weightskewed (self, main, norm):
        return norm.weightskewed (main, self.subgroup)

    def __check_sym (self, order):
        """Checks the availability of the symbolic solver."""
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

    def __get_enforce_vec (self, enforce_index, norm, optimizer):
        """Interprets the enforce specification."""
        if optimizer == "main":
            if enforce_index == 0: #pending implementation
                raise ValueError ("this functionality is removed due to support for subgroup tuning. ")
            else:
                return te.monzo2ratio (self.subgroup.basis_matrix.T[enforce_index - 1], self.subgroup)
        elif optimizer == "sym":
            if enforce_index == 0: #pending implementation
                raise ValueError ("this functionality is removed due to support for subgroup tuning. ")
            else:
                return te.monzo2ratio (self.subgroup.basis_matrix.T[enforce_index - 1], self.subgroup)

    def __mean (self, main):
        """
        This mean rejects the extra dimension from the denominator
        such that when skew = 0, introducing the extra dimension doesn't change the result.
        """
        return np.sum (main)/len (self.subgroup)

    def __power_mean_norm (self, main, order):
        if order == np.inf:
            return np.max (main)
        else:
            return np.power (self.__mean (np.power (np.abs (main), order)), np.reciprocal (float (order)))

    def __show_header (self, norm = None, enforce_text = None, ntype = None):
        print (f"\nSubgroup: {self.subgroup}",
            f"Mapping: \n{self.vals}", sep = "\n")

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
            constraint = None, destretch = None, *, 
            enforce = None, cons_monzo_list = None, des_monzo = None): #deprecated parameters

        if not enforce is None:
            warnings.warn ("\"enforce\" is deprecated. Use \"constraint\" and/or \"destretch\" instead. ")
            if enforce == "c": #default to octave-constrained
                enforce = "c1"
            elif enforce == "d": #default to octave-destretched
                enforce = "d1"
            if constraint is None and destretch is None:
                if enforce_spec_list := re.findall ("[cd]\d+", str (enforce)): #separates the enforcements
                    cons_ratios, des_ratios, cons_text, des_text = ([] for _ in range (4))
                    for entry in enforce_spec_list:
                        if entry[0] == "c":
                            cons_ratios.append (self.__get_enforce_vec (int (entry[1:]), norm, optimizer))
                        else:
                            des_ratios.append (self.__get_enforce_vec (int (entry[1:]), norm, optimizer))
                    
                    constraint = te.Subgroup (cons_ratios) if cons_ratios else None
                    destretch = te.as_ratio (*des_ratios) if des_ratios else None

        if not cons_monzo_list is None:
            constraint = te.Subgroup ([te.monzo2ratio (entry) for entry in cons_monzo_list.T])
            warnings.warn ("\"cons_monzo_list\" is deprecated. Use \"constraint\" instead. ")
        if not des_monzo is None:
            destretch = te.monzo2ratio (des_monzo)
            warnings.warn ("\"des_monzo\" is deprecated. Use \"destretch\" instead. ")

        # checks optimizer availability
        if optimizer == "sym" and not self.__check_sym (norm.order):
            return self.tune (optimizer = "main", norm = norm,
                constraint = constraint, destretch = destretch)

        # gets the enforcement text
        cons_text = constraint.__str__ () + "-constrained" if constraint else ""
        des_text = destretch.__str__ () + "-destretched" if destretch else ""
        enforce_text = " ".join ([cons_text, des_text]) if cons_text or des_text else "none"

        # shows the header
        self.__show_header (norm = norm, enforce_text = enforce_text)

        # optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tempered_tuning_map, mistuning_map = te_opt.optimizer_main (
                self.vals, target = self.subgroup, norm = norm, 
                constraint = constraint, destretch = destretch
            )
        elif optimizer == "sym":
            gen, tempered_tuning_map, mistuning_map = te_sym.symbolic (
                self.vals, target = self.subgroup, norm = te_sym.NormSym (norm), 
                constraint = constraint, destretch = destretch
            )

        # error and bias
        tempered_tuning_map_x = self.weightskewed (tempered_tuning_map, norm)
        mistuning_map_x = self.weightskewed (mistuning_map, norm)
        error = self.__power_mean_norm (mistuning_map_x, norm.order)
        bias = np.mean (mistuning_map_x)
        # print (mistuning_map_x) #for debugging
        print (f"Tuning error: {error:.6f} (¢)",
            f"Tuning bias: {bias:.6f} (¢)", sep = "\n")
        return gen, tempered_tuning_map, mistuning_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, norm = te.Norm (), show = True):
        combination_list = itertools.combinations (range (self.vals.shape[1]), self.vals.shape[0])
        wedgie = np.array ([linalg.det (self.weightskewed (self.vals, norm)[:, entry]) for entry in combination_list])
        wedgie *= np.copysign (1, wedgie[0])
        if show:
            self.__show_header ()
            print (f"Wedgie: {wedgie}", sep = "\n")
        return wedgie

    def complexity (self, ntype = "breed", norm = te.Norm ()):
        if norm.order != 2:
            raise NotImplementedError ("temperament measures only work for Euclidean norm as of now. ")
        
        # standard L2 complexity
        complexity = np.sqrt (linalg.det (
            self.weightskewed (self.vals, norm)
            @ self.weightskewed (self.vals, norm).T))
        # complexity = linalg.norm (self.wedgie (norm = norm, show = False)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.vals.shape[1]**self.vals.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (tuple (itertools.combinations (range (self.vals.shape[1]), self.vals.shape[0]))))
        elif ntype == "none":
            pass
        else:
            warnings.warn ("normalizer not supported, using default (\"breed\")")
            return self.complexity (ntype = "breed", norm = norm)
        return complexity

    def error (self, ntype = "breed", norm = te.Norm (), scalar = te.SCALAR.CENT): #in cents by default
        if norm.order != 2:
            raise NotImplementedError ("temperament measures only work for Euclidean norm as of now. ")

        # standard L2 error
        error = linalg.norm (
            self.weightskewed (self.subgroup.just_tuning_map (scalar), norm)
            @ linalg.pinv (self.weightskewed (self.vals, norm))
            @ self.weightskewed (self.vals, norm)
            - self.weightskewed (self.subgroup.just_tuning_map (scalar), norm))
        if ntype == "breed": #Graham Breed's RMS (default)
            error *= 1/np.sqrt (self.vals.shape[1])
        elif ntype == "smith": #Gene Ward Smith's RMS
            try:
                error *= np.sqrt ((self.vals.shape[0] + 1) / (self.vals.shape[1] - self.vals.shape[0]))
            except ZeroDivisionError:
                error = np.nan
        elif ntype == "none":
            pass
        else:
            warnings.warn ("normalizer not supported, using default (\"breed\")")
            return self.error (ntype = "breed", norm = norm)
        return error

    def badness (self, ntype = "breed", norm = te.Norm (), scalar = te.SCALAR.OCTAVE): #in octaves by default
        return (self.error (ntype, norm, scalar)
            * self.complexity (ntype, norm))

    def badness_logflat (self, ntype = "breed", norm = te.Norm (), scalar = te.SCALAR.OCTAVE): #in octaves
        try:
            return (self.error (ntype, norm, scalar)
                * self.complexity (ntype, norm)**(self.vals.shape[1]/(self.vals.shape[1] - self.vals.shape[0])))
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", norm = te.Norm (), badness_scale = 1000):
        self.__show_header (norm = norm, ntype = ntype)

        # shows the temperament measures
        error = self.error (ntype, norm)
        complexity = self.complexity (ntype, norm)
        badness = self.badness (ntype, norm) * badness_scale
        badness_logflat = self.badness_logflat (ntype, norm) * badness_scale
        print (f"Complexity: {complexity:.6f}",
            f"Error: {error:.6f} (¢)",
            f"Badness (simple): {badness:.6f} (oct/{badness_scale})",
            f"Badness (logflat): {badness_logflat:.6f} (oct/{badness_scale})", sep = "\n")

    def comma_basis (self, show = True):
        comma_basis_frac = Matrix (self.vals).nullspace ()
        comma_basis = np.column_stack ([te.matrix2array (entry) for entry in comma_basis_frac])
        if show:
            self.__show_header ()
            print ("Comma basis: ")
            te.show_monzo_list (comma_basis_frac, self.subgroup)
        return comma_basis
