# © 2020-2024 Flora Canou | Version 1.6.3
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te

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
            if norm.wtype == "tenney" and norm.skew == 1:
                weightskew_text = "weil"
            else:
                weight_text = norm.wtype
                if norm.wamount != 1:
                    weight_text += f"[{norm.wamount}]"
                if norm.skew != 0:
                    skew_text = "skewed"
                    if norm.skew != 1:
                        skew_text += f"[{norm.skew}]"
                    weightskew_text = "-".join ((skew_text, weight_text))
                else:
                    weightskew_text = weight_text

            order_dict = {1: "chebyshevian", 2: "euclidean", np.inf: "manhattan"}
            try:
                order_text = order_dict[norm.order]
            except KeyError:
                order_text = f"L{norm.order}"

            print ("Norm: " + "-".join ((weightskew_text, order_text)))
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
        if is_trivial := (self.subgroup.is_prime ()
                or norm.wtype == "tenney" and self.subgroup.is_prime_power ()):
            mode_text = "trivial -- inharmonic and subgroup tunings are identical"
        else:
            mode_text = "inharmonic tuning" if inharmonic else "subgroup tuning"

        # shows the header
        self.__show_header (norm = norm, mode_text = mode_text, enforce_text = enforce_text)

        # optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tempered_tuning_map, error_map = te_opt.wrapper_main (
                self.mapping, subgroup = self.subgroup, norm = norm, 
                inharmonic = inharmonic or is_trivial, constraint = constraint, destretch = destretch
            )
        elif optimizer == "sym":
            gen, tempered_tuning_map, error_map = te_sym.wrapper_symbolic (
                self.mapping, subgroup = self.subgroup, norm = te_sym.NormSym (norm), 
                inharmonic = inharmonic or is_trivial, constraint = constraint, destretch = destretch
            )

        return gen, tempered_tuning_map, error_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, norm = te.Norm (wtype = "equilateral"), show = True):
        combinations = itertools.combinations (
            range (self.mapping.shape[1] + (1 if norm.skew else 0)), self.mapping.shape[0])
        wedgie = np.array (
            [linalg.det (norm.tuning_x (self.mapping, self.subgroup)[:, entry]) for entry in combinations])
        wedgie *= np.copysign (1, wedgie[0])

        # convert to integer type if possible
        wedgie_int = wedgie.astype (int)
        if all (wedgie == wedgie_int):
            wedgie = wedgie_int
        
        if show:
            self.__show_header ()
            print (f"Wedgie: {wedgie}", sep = "\n")
        
        return wedgie

    def complexity (self, ntype = "breed", norm = te.Norm (), inharmonic = False):
        """
        Returns the temperament's complexity, 
        nondegenerate subgroup temperaments supported. 
        """
        if not norm.order == 2:
            raise ValueError ("this measure is only defined on Euclidean norms. ")
        do_inharmonic = (inharmonic or self.subgroup.is_prime ()
            or norm.wtype == "tenney" and self.subgroup.is_prime_power ())
        if not do_inharmonic and self.subgroup.index () == np.inf:
            raise ValueError ("this measure is only defined on nondegenerate subgroups. ")
        return self.__complexity (ntype, norm, inharmonic = do_inharmonic)

    def error (self, ntype = "breed", norm = te.Norm (), inharmonic = False, scalar = te.SCALAR.CENT): #in cents by default
        """
        Returns the temperament's inherent inaccuracy regardless of the actual tuning, 
        all subgroup temperaments supported. 
        """
        if not norm.order == 2:
            raise NotImplementedError ("non-Euclidean norms not supported as of now. ")
        do_inharmonic = (inharmonic or self.subgroup.is_prime ()
            or norm.wtype == "tenney" and self.subgroup.is_prime_power ())
        return self.__error (ntype, norm, inharmonic = do_inharmonic, scalar = scalar)

    def __complexity (self, ntype, norm, inharmonic):
        if inharmonic:
            subgroup = self.subgroup
            mapping = self.mapping
            index = 1
        else:
            subgroup = self.subgroup.minimal_prime_subgroup ()
            mapping = te.antinullspace (self.subgroup.basis_matrix_to (subgroup) @ te.nullspace (self.mapping))
            index = self.subgroup.index ()
        r, d = mapping.shape #rank and dimensionality

        # standard L2 complexity
        complexity = np.sqrt (linalg.det (
            norm.tuning_x (mapping, subgroup)
            @ norm.tuning_x (mapping, subgroup).T
            )) / index
        # complexity = linalg.norm (self.wedgie (norm = norm)) #same
        match ntype:
            case "breed": #Graham Breed's RMS (default)
                complexity *= 1/np.sqrt (d**r)
            case "smith": #Gene Ward Smith's RMS
                complexity *= 1/np.sqrt (len (tuple (itertools.combinations (range (d), r))))
            case "dirichlet": #Sintel's Dirichlet coefficient
                complexity *= 1/linalg.det (norm.tuning_x (np.eye (d), subgroup)[:,:d])**(r/d)
            case "none":
                pass
            case _:
                warnings.warn ("normalizer not supported, using default (\"breed\")")
                return self.__complexity (ntype = "breed", norm = norm, inharmonic = inharmonic)
        return complexity

    def __error (self, ntype, norm, inharmonic, scalar):
        if inharmonic:
            subgroup = self.subgroup
            mapping = self.mapping
        else:
            subgroup = self.subgroup.minimal_prime_subgroup ()
            mapping = te.antinullspace (self.subgroup.basis_matrix_to (subgroup) @ te.nullspace (self.mapping))
        r, d = mapping.shape #rank and dimensionality
        just_tuning_map = subgroup.just_tuning_map (scalar)

        # standard L2 error
        error = linalg.norm (
            norm.tuning_x (just_tuning_map, subgroup)
            @ linalg.pinv (norm.tuning_x (mapping, subgroup))
            @ norm.tuning_x (mapping, subgroup)
            - norm.tuning_x (just_tuning_map, subgroup)
        )
        match ntype: 
            case "breed": #Graham Breed's RMS (default)
                error *= 1/np.sqrt (d)
            case "smith": #Gene Ward Smith's RMS
                try:
                    error *= np.sqrt ((r + 1)/(d - r))
                except ZeroDivisionError:
                    error = np.nan
            case "dirichlet": #Sintel's Dirichlet coefficient
                # Dirichlet--Breed, actually
                # as an extra factor of 1/(np.sqrt (d) is added here
                # which isn't in Sintel's implementation
                # this factor will be canceled out in logflat badness
                # when we divide it by the norm of jtm
                error *= 1/(np.sqrt (d)
                    * linalg.det (norm.tuning_x (np.eye (d), subgroup)[:,:d])**(1/d))
            case "none":
                pass
            case _:
                warnings.warn ("normalizer not supported, using default (\"breed\")")
                return self.__error ("breed", norm, inharmonic, scalar)
        return error

    def badness (self, ntype = "breed", norm = te.Norm (), inharmonic = False, 
            logflat = False, scalar = te.SCALAR.OCTAVE): #in octaves by default
        if not norm.order == 2:
            raise ValueError ("this measure is only defined on Euclidean norms. ")
        do_inharmonic = (inharmonic or self.subgroup.is_prime ()
            or norm.wtype == "tenney" and self.subgroup.is_prime_power ())
        if not do_inharmonic and self.subgroup.index () == np.inf:
            raise ValueError ("this measure is only defined on nondegenerate subgroups. ")

        if logflat:
            return self.__badness_logflat (ntype, norm, inharmonic, scalar)
        else:
            return self.__badness (ntype, norm, inharmonic, scalar)

    def __badness (self, ntype, norm, inharmonic, scalar):
        return (self.__error (ntype, norm, inharmonic, scalar)
            * self.__complexity (ntype, norm, inharmonic))

    def __badness_logflat (self, ntype, norm, inharmonic, scalar):
        r, d = self.mapping.shape #rank and dimensionality
        try:
            res = (self.__error (ntype, norm, inharmonic, scalar)
                * self.__complexity (ntype, norm, inharmonic)**(d/(d - r)))
        except ZeroDivisionError:
            res = np.nan
        match ntype:
            case "dirichlet":
                norm_jtm = 1/linalg.det (norm.tuning_x (np.eye (d), self.subgroup)[:,:d])**(1/d)
                res /= norm_jtm
            case "breed" | "smith" | "none":
                pass
            case _:
                warnings.warn ("normalizer not supported, using default (\"breed\")")
                return self.__badness_logflat ("breed", norm, inharmonic, scalar)
        return res

    def temperament_measures (self, ntype = "breed", norm = te.Norm (), inharmonic = False, 
            error_scale = te.SCALAR.CENT, badness_scale = te.SCALAR.OCTAVE):
        """Shows the temperament measures."""
        if not norm.order == 2:
            raise ValueError ("this measure is only defined on Euclidean norms. ")
        do_inharmonic = (inharmonic or self.subgroup.is_prime ()
            or norm.wtype == "tenney" and self.subgroup.is_prime_power ())
        if not do_inharmonic and self.subgroup.index () == np.inf:
            raise ValueError ("this measure is only defined on nondegenerate subgroups. ")
        return self.__temperament_measures (ntype, norm, do_inharmonic, error_scale, badness_scale)
        
    def __temperament_measures (self, ntype, norm, inharmonic, error_scale, badness_scale):
        self.__show_header (norm = norm, ntype = ntype)
        complexity = self.__complexity (ntype, norm, inharmonic)
        error = self.__error (ntype, norm, inharmonic, error_scale)
        badness = self.__badness (ntype, norm, inharmonic, badness_scale)
        badness_logflat = self.__badness_logflat (ntype, norm, inharmonic, badness_scale)
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
