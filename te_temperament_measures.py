# © 2020-2025 Flora Canou
# This work is licensed under the GNU General Public License version 3.

import itertools, re, warnings
import numpy as np
from scipy import linalg
from sympy.matrices import Matrix, BlockMatrix
from sympy import gcd
import te_common as te

class Temperament:
    # NOTE: "map" is a reserved word

    def __init__ (self, breeds, subgroup = None, *, saturate = True, normalize = True):
        breeds, subgroup = te.setup (breeds, subgroup, axis = te.AXIS.ROW)
        self.subgroup = subgroup
        self.mapping = te.canonicalize (np.rint (breeds).astype (int), saturate, normalize)

    @staticmethod
    def __check_sym (order):
        """Checks the applicability and availability of the symbolic solver."""
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

    def form (self, ftype = "none"):
        """
        Returns the mapping renormalized to various forms. 
        \"none\": does nothing. 
        \"flip\": flips negative generators. 
        \"shift\": 
            flips negative generators if they are (-1)-sheared; 
            equave-reduces negative generators otherwise. 
        \"reduce\": 
            flips negative generators if they are (-1)-sheared; 
            equave-reduces all generators. 
        """

        def __fx ():
            """
            Returns the fast approximate generator map in octaves.
            The mapping must be in HNF. 
            """
            cols = [next (j for j, bj in enumerate (self.mapping[i]) if bj != 0) for i in range (self.mapping.shape[0])]
            return self.subgroup.just_tuning_map ()[cols] @ linalg.inv (self.mapping[:, cols])

        mapping = self.mapping.copy ()
        match ftype:
            case "none":
                pass
            case "flip":
                gen = __fx ()
                for i, gi in enumerate (gen):
                    if gi < 0:
                        mapping[i] *= -1
            case "shift":
                # to determine when to flip
                # we define a generalized ploidacot that works for any rank & subgroup
                # ploid: number of periods per formal equave
                #   which equals the first entry of the mapping
                # cot: number of gens to reach the interval indicated by the current row
                #   which equals the first nonzero entry of the current row
                # shear: number of whole periods the interval has, modulo cot
                # to perform equave reduction on the mapping
                # we add to the first row the current row times the number
                # of whole periods of the number of whole equaves the gen has
                gen = __fx ()
                ploid = mapping[0][0]
                for i, gi in enumerate (gen[1:], start = 1):
                    if gi < 0:
                        cot = next (entry for entry in mapping[i] if entry != 0)
                        shear = np.floor (cot*gi/gen[0]).astype (int) % cot
                        if shear == cot - 1: 
                            mapping[i] *= -1
                        else:
                            mapping[0] += mapping[i]*ploid*np.floor (gi/(ploid*gen[0])).astype (int)
            case "reduce":
                # similar to above, but equave reduction is done on every row
                gen = __fx ()
                ploid = mapping[0][0]
                for i, gi in enumerate (gen[1:], start = 1):
                    if gi < 0:
                        cot = next (entry for entry in mapping[i] if entry != 0)
                        shear = np.floor (cot*gi/gen[0]).astype (int) % cot
                        if shear == cot - 1: 
                            mapping[i] *= -1
                            gi *= -1 #updates the gen for use below
                    mapping[0] += mapping[i]*ploid*np.floor (gi/(ploid*gen[0])).astype (int)
            case _:
                warnings.warn ("form not supported, using default (\"none\"). ")
                return self.form ("none")
        return mapping

    def tune (self, optimizer = "main", norm = te.Norm (), inharmonic = False, 
            constraint = None, destretch = None, ftype = None): 
        """
        Gives the tuning. 
        Calls either wrapper_main or wrapper_symbolic. 
        """

        # checks optimizer applicability and availability
        if optimizer == "sym" and not self.__check_sym (norm.order):
            return self.tune (optimizer = "main", norm = norm, inharmonic = inharmonic, 
                constraint = constraint, destretch = destretch, ftype = ftype)

        # gets the text for enforcement & mode
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

        # gets the specified form
        if ftype:
            mapping = self.form (ftype)
            print ("The generators will be found for the following mapping form:", 
                mapping, sep = "\n")
        else:
            mapping = self.mapping

        # starts optimization
        if optimizer == "main":
            import te_optimizer as te_opt
            gen, tempered_tuning_map, error_map = te_opt.wrapper_main (
                mapping, subgroup = self.subgroup, norm = norm, 
                inharmonic = inharmonic or is_trivial, constraint = constraint, destretch = destretch
            )
        elif optimizer == "sym":
            gen, tempered_tuning_map, error_map = te_sym.wrapper_symbolic (
                mapping, subgroup = self.subgroup, norm = te_sym.NormSym (norm), 
                inharmonic = inharmonic or is_trivial, constraint = constraint, destretch = destretch
            )

        return gen, tempered_tuning_map, error_map

    optimise = tune
    optimize = tune
    analyze = tune
    analyse = tune

    def wedgie (self, norm = te.Norm (wtype = "equilateral"), show = True):
        """Finds the wedgie of the temperament. """
        wedgie = self.__wedgie (self.mapping, self.subgroup, norm)

        # normalize for a positive first entry
        # unneeded if the mapping is in canonical form
        if wedgie[0] < 0:
            wedgie *= -1

        # convert to integer type if possible
        wedgie_rd = np.rint (wedgie)
        if np.allclose (wedgie, wedgie_rd, rtol = 0, atol = 1e-6):
            wedgie = wedgie_rd.astype (int)
        
        if show:
            self.__show_header ()
            print (f"Wedgie: {wedgie}", sep = "\n")
        
        return wedgie

    @staticmethod
    def __wedgie (mapping, subgroup, norm): 
        r, d = mapping.shape #rank and dimensionality
        if norm.skew: d += 1
        combinations = itertools.combinations (range (d), r)
        return np.array ([linalg.det (norm.tuning_x (mapping, subgroup)[:, entry]) for entry in combinations])

    def complexity (self, ntype = "breed", norm = te.Norm (), inharmonic = False):
        """
        Returns the temperament's complexity, 
        nondegenerate subgroup temperaments supported. 
        """
        do_inharmonic = (inharmonic or self.subgroup.is_prime ()
            or norm.wtype == "tenney" and self.subgroup.is_prime_power ())
        if not do_inharmonic and self.subgroup.index () == np.inf:
            raise ValueError ("this measure is only defined on nondegenerate subgroups. ")
        return self.__complexity (ntype, norm, do_inharmonic)

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

        if norm.order == 2: # standard L2 complexity
            # complexity = linalg.norm (
            #     self.__wedgie (mapping, subgroup, norm)) / index #same but less performant
            complexity = np.sqrt (linalg.det (
                norm.tuning_x (mapping, subgroup)
                @ norm.tuning_x (mapping, subgroup).T)) / index
        else:
            complexity = linalg.norm (
                self.__wedgie (mapping, subgroup, norm), ord = norm.order) / index
        
        match ntype:
            case "breed": #Graham Breed's RMS (default)
                complexity *= 1/(d**r)**(1/norm.order)
            case "smith": #Gene Ward Smith's RMS
                complexity *= 1/(len (tuple (itertools.combinations (range (d), r))))**(1/norm.order)
            case "sintel": #Sintel--Breed
                complexity *= 1/linalg.det (norm.tuning_x (np.eye (d), subgroup)[:,:d])**(r/d)
            case "none":
                pass
            case _:
                warnings.warn ("normalizer not supported, using default (\"breed\"). ")
                return self.__complexity ("breed", norm, inharmonic)
        
        return complexity

    def error (self, ntype = "breed", norm = te.Norm (), inharmonic = False, scalar = te.SCALAR.CENT): #in cents by default
        """
        Returns the temperament's inherent inaccuracy regardless of the actual tuning, 
        all subgroup temperaments supported. 
        """
        do_inharmonic = (inharmonic or self.subgroup.is_prime ()
            or norm.wtype == "tenney" and self.subgroup.is_prime_power ())
        return self.__error (ntype, norm, do_inharmonic, scalar)

    def __error (self, ntype, norm, inharmonic, scalar):
        if inharmonic:
            subgroup = self.subgroup
            mapping = self.mapping
        else:
            subgroup = self.subgroup.minimal_prime_subgroup ()
            mapping = te.antinullspace (self.subgroup.basis_matrix_to (subgroup) @ te.nullspace (self.mapping))
        r, d = mapping.shape #rank and dimensionality
        just_tuning_map = subgroup.just_tuning_map (scalar)

        if norm.order == 2: # standard L2 error
            error_map_x = (norm.tuning_x (just_tuning_map, subgroup)
                @ linalg.pinv (norm.tuning_x (mapping, subgroup))
                @ norm.tuning_x (mapping, subgroup)
                - norm.tuning_x (just_tuning_map, subgroup))
            # error = linalg.norm (error_map_x) #same
            error = np.sqrt (error_map_x @ error_map_x.T)
        else:
            import te_optimizer as te_opt
            _, _, error_map = te_opt.optimizer_main (
                mapping, target = subgroup, norm = norm, show = False)
            error_map *= scalar / te.SCALAR.CENT #optimizer is always in cents
            error_map_x = norm.tuning_x (error_map, subgroup)
            error = linalg.norm (error_map_x, ord = norm.order)
        
        match ntype: 
            case "breed": #Graham Breed's RMS (default)
                error *= 1/d**(1/norm.order)
            case "smith": #Gene Ward Smith's RMS
                try:
                    error *= ((r + 1)/(d - r))**(1/norm.order)
                except ZeroDivisionError:
                    error = np.nan
            case "sintel": #Sintel--Breed
                # an extra factor of 1/(np.sqrt (d)) is added here
                # which isn't in Sintel's implementation
                # this factor will be canceled out in logflat badness
                # when we divide it by the norm of jtm
                error *= 1/(d**(1/norm.order)
                    * linalg.det (norm.tuning_x (np.eye (d), subgroup)[:,:d])**(1/d))
            case "none":
                pass
            case _:
                warnings.warn ("normalizer not supported, using default (\"breed\"). ")
                return self.__error ("breed", norm, inharmonic, scalar)
        return error

    def badness (self, ntype = "breed", norm = te.Norm (), inharmonic = False, 
            logflat = False, scalar = te.SCALAR.OCTAVE): #in octaves by default

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
            case "sintel":
                norm_jtm = 1/linalg.det (norm.tuning_x (np.eye (d), self.subgroup)[:,:d])**(1/d)
                res /= norm_jtm
            case "breed" | "smith" | "none":
                pass
            case _:
                warnings.warn ("normalizer not supported, using default (\"breed\"). ")
                return self.__badness_logflat ("breed", norm, inharmonic, scalar)
        return res

    def temperament_measures (self, ntype = "breed", norm = te.Norm (), inharmonic = False, 
            error_scale = te.SCALAR.CENT, badness_scale = te.SCALAR.OCTAVE):
        """Shows the temperament measures."""
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
