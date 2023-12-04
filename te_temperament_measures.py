# © 2020-2023 Flora Canou | Version 1.1.0
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te
np.set_printoptions (suppress = True, linewidth = 256, precision = 4)

class Temperament:
    def __init__ (self, breeds, subgroup = None, saturate = True, normalize = True): #NOTE: "map" is a reserved word
        breeds, subgroup = te.setup (breeds, subgroup, axis = te.AXIS.ROW)
        self.subgroup = subgroup
        self.mapping = te.canonicalize (np.rint (breeds).astype (int), saturate, normalize)

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

    def __show_header (self, norm = None, mode_text = None, enforce_text = None, ntype = None):
        print (f"\nSubgroup: {self.subgroup}",
            f"Mapping: \n{self.mapping}", sep = "\n")

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

            print ("Norm: " + weight_text + skew_text + order_text)
        if mode_text:
            print ("Mode: " + mode_text)
        if enforce_text:
            print ("Enforcement: " + enforce_text)
        if ntype:
            print ("Normalizer: " + ntype)
        return

    def tune (self, optimizer = "main", norm = te.Norm (), inharmonic = False, 
            constraint = None, destretch = None): 
        """
        Gives the tuning. 
        Calls either wrapper_main or wrapper_symbolic. 
        """

        # checks optimizer availability
        if optimizer == "sym" and not self.__check_sym (norm.order):
            return self.tune (optimizer = "main", norm = norm, inharmonic = inharmonic, 
                constraint = constraint, destretch = destretch)

        # gets the enforcement text
        cons_text = constraint.__str__ () + "-constrained" if constraint else ""
        des_text = destretch.__str__ () + "-destretched" if destretch else ""
        enforce_text = " ".join ([cons_text, des_text]) if cons_text or des_text else "none"
        if is_trivial := (self.subgroup.is_trivial ()
                or norm.wtype == "tenney" and subgroup.is_tenney_trivial ()):
            mode_text = "prime-harmonic"
        else:
            mode_text = "inharmonic" if inharmonic else "subgroup"

        # shows the header
        self.__show_header (norm = norm, mode_text = mode_text, enforce_text = enforce_text)

        # optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tempered_tuning_map, error_map = te_opt.wrapper_main (
                self.mapping, subgroup = self.subgroup, norm = norm, inharmonic = is_trivial, 
                constraint = constraint, destretch = destretch
            )
        elif optimizer == "sym":
            gen, tempered_tuning_map, error_map = te_sym.wrapper_symbolic (
                self.mapping, subgroup = self.subgroup, norm = te_sym.NormSym (norm), inharmonic = is_trivial, 
                constraint = constraint, destretch = destretch
            )

        return gen, tempered_tuning_map, error_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, norm = te.Norm (), show = True):
        combinations = itertools.combinations (range (self.mapping.shape[1]), self.mapping.shape[0])
        wedgie = np.array ([
            linalg.det (norm.tuning_x (self.mapping, self.subgroup)[:, entry]) for entry in combinations
        ])
        wedgie *= np.copysign (1, wedgie[0])
        if show:
            self.__show_header ()
            print (f"Wedgie: {wedgie}", sep = "\n")
        return wedgie

    def complexity (self, ntype = "breed", norm = te.Norm ()):
        if not norm.order == 2:
            raise NotImplementedError ("non-Euclidean norms not supported as of now. ")
        elif not (self.subgroup.is_trivial ()
                or norm.wtype == "tenney" and self.subgroup.is_tenney_trivial ()):
            raise NotImplementedError ("nontrivial subgroups not supported as of now. ")
        return self.__complexity (ntype, norm)

    def error (self, ntype = "breed", norm = te.Norm (), inharmonic = False, scalar = te.SCALAR.CENT): #in cents by default
        """
        Returns the temperament's inherent inaccuracy regardless of the actual tuning, 
        subgroup temperaments supported. 
        """
        if not norm.order == 2:
            raise NotImplementedError ("non-Euclidean norms not supported as of now. ")
        is_trivial = (inharmonic or self.subgroup.is_trivial ()
            or norm.wtype == "tenney" and self.subgroup.is_tenney_trivial ())
        return self.__error (ntype, norm, inharmonic = is_trivial, scalar = scalar)

    def __complexity (self, ntype, norm):
        # standard L2 complexity
        complexity = np.sqrt (linalg.det (
            norm.tuning_x (self.mapping, self.subgroup)
            @ norm.tuning_x (self.mapping, self.subgroup).T
        ))
        # complexity = linalg.norm (self.wedgie (norm = norm, show = False)) #same
        if ntype == "breed": #Graham Breed's RMS (default)
            complexity *= 1/np.sqrt (self.mapping.shape[1]**self.mapping.shape[0])
        elif ntype == "smith": #Gene Ward Smith's RMS
            complexity *= 1/np.sqrt (len (tuple (itertools.combinations (range (self.mapping.shape[1]), self.mapping.shape[0]))))
        elif ntype == "none":
            pass
        else:
            warnings.warn ("normalizer not supported, using default (\"breed\")")
            return self.__complexity (ntype = "breed", norm = norm)
        return complexity

    def __error (self, ntype, norm, inharmonic, scalar):
        # standard L2 error
        if inharmonic:
            mapping = self.mapping
            subgroup = self.subgroup
        else:
            mapping = te.antinullspace (self.subgroup.basis_matrix @ te.nullspace (self.mapping))
            subgroup = te.get_subgroup (self.subgroup.basis_matrix, axis = te.AXIS.COL)
        just_tuning_map = subgroup.just_tuning_map (scalar)
        error = linalg.norm (
            norm.tuning_x (just_tuning_map, subgroup)
            @ linalg.pinv (norm.tuning_x (mapping, subgroup))
            @ norm.tuning_x (mapping, subgroup)
            - norm.tuning_x (just_tuning_map, subgroup)
        )
        if ntype == "breed": #Graham Breed's RMS (default)
            error *= 1/np.sqrt (mapping.shape[1])
        elif ntype == "smith": #Gene Ward Smith's RMS
            try:
                error *= np.sqrt ((mapping.shape[0] + 1) / (mapping.shape[1] - mapping.shape[0]))
            except ZeroDivisionError:
                error = np.nan
        elif ntype == "none":
            pass
        else:
            warnings.warn ("normalizer not supported, using default (\"breed\")")
            return self.__error (ntype = "breed", norm = norm, inharmonic = inharmonic, scalar = scalar)
        return error

    def badness (self, ntype = "breed", norm = te.Norm (), logflat = False, scalar = te.SCALAR.OCTAVE): #in octaves by default
        if not norm.order == 2:
            raise NotImplementedError ("non-Euclidean norms not supported as of now. ")
        elif not (self.subgroup.is_trivial ()
                or norm.wtype == "tenney" and self.subgroup.is_tenney_trivial ()):
            raise NotImplementedError ("nontrivial subgroups not supported as of now. ")

        if logflat:
            return __badness_logflat (ntype, norm, scalar)
        else:
            return __badness (ntype, norm, scalar)

    def __badness (self, ntype, norm, scalar):
        return (self.__error (ntype, norm, inharmonic = True, scalar = scalar)
            * self.__complexity (ntype, norm))

    def __badness_logflat (self, ntype, norm, scalar):
        try:
            return (self.__error (ntype, norm, inharmonic = True, scalar = scalar)
                * self.__complexity (ntype, norm)**(self.mapping.shape[1]/(self.mapping.shape[1] - self.mapping.shape[0])))
        except ZeroDivisionError:
            return np.nan

    def temperament_measures (self, ntype = "breed", norm = te.Norm (), error_scale = te.SCALAR.CENT, badness_scale = 1e3):
        """Shows the temperament measures."""
        if not norm.order == 2:
            raise NotImplementedError ("non-Euclidean norms not supported as of now. ")
        elif not (self.subgroup.is_trivial ()
                or norm.wtype == "tenney" and self.subgroup.is_tenney_trivial ()):
            raise NotImplementedError ("nontrivial subgroups not supported as of now. ")
        return self.__temperament_measures (ntype, norm, error_scale, badness_scale)
        
    def __temperament_measures (self, ntype, norm, error_scale, badness_scale):
        self.__show_header (norm = norm, ntype = ntype)
        error = self.__error (ntype, norm, inharmonic = False, scalar = error_scale)
        complexity = self.__complexity (ntype, norm)
        badness = self.__badness (ntype, norm, scalar = badness_scale)
        badness_logflat = self.__badness_logflat (ntype, norm, scalar = badness_scale)
        print (f"Complexity: {complexity:.6f}",
            f"Error: {error:.6f} (oct/{error_scale})",
            f"Badness (simple): {badness:.6f} (oct/{badness_scale})",
            f"Badness (logflat): {badness_logflat:.6f} (oct/{badness_scale})", sep = "\n")

    def comma_basis (self, show = True):
        comma_basis = te.nullspace (self.mapping)
        if show:
            self.__show_header ()
            print ("Comma basis: ")
            te.show_monzo_list (comma_basis, self.subgroup)
        return comma_basis
