# Temperament Evaluator

## Dependencies

- [SciPy](https://scipy.org/)
	- various tasks including optimization, temperament measures, wedgie, etc. 
- [SymPy](https://www.sympy.org/en/index.html)
	- various tasks including symbolic solution, mapping normalization, comma basis, etc. 

## `te_common.py`

Common functions. Required by virtually all subsequent modules. 

Use the `Subgroup` class to create a just intonation subgroup. Parameters: 
- `ratios`: list of ratios, fractional notation supported. 
- `monzos`: matrix of monzos, alternative way to initialize it. 

Use the `Norm` class to create a norm profile for the tuning space. Parameters: 
- `wtype`: Weight method. Has `"tenney"` (default), `"equilateral"`, and `"wilson"`/`"benedetti"`. 
- `wamount`: Weight scaling factor. Default is `1`. 
- `skew`: Skew. This is Mike Battaglia's *k*. Default is `0`, meaning no skew. For **Weil**, use `1`. 
- `order`: Order. Default is `2`, meaning **Euclidean**. For **XOP tuning**, use `np.inf`. 

## `te_optimizer.py`

Optimizes tunings. Custom norm profile, constraints and destretch are supported. *It is recommended to use `te_temperament_measures` instead since it calls this module and it has a more accessible interface.*

Requires `te_common`. 

Use `wrapper_main` to optimize a temperament. Parameters: 
- `breeds`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Custom subgroup for the map. Default is prime harmonics. 
- `norm`: *optional*. Norm profile for the tuning space. See above. 
- `inharmonic`: *optional*. For subgroup temperaments, treats the basis as if they were primes. Default is `False`. 
- `constraint`: *optional*. Constrains this subgroup to pure. Default is empty. 
- `destretch`: *optional*. Destretches this ratio to pure. Default is empty. 
- `show`: *optional*. Displays the result. Default is `True`. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as an array of column vectors.** 

## `te_optimizer_legacy.py`

Legacy single-file edition (doesn't require `te_common.py`). Support for subgroup tuning is limited. 

## `te_symbolic.py`

Solves Euclidean tunings symbolically. *It is recommended to use `te_temperament_measures` instead since it calls this module and it has a more accessible interface.*

Requires `te_common`. 

Use `wrapper_symbolic` to solve for a Euclidean tuning of a temperament. Parameters: 
- `breeds`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Custom subgroup for the map. Default is prime harmonics. 
- `norm`: *optional*. Norm profile for the tuning space. See above. 
- `inharmonic`: *optional*. For subgroup temperaments, treats the basis as if they were primes. Default is `False`. 
- `constraint`: *optional*. Constrains this subgroup to pure. Default is empty. 
- `destretch`: *optional*. Destretches this ratio to pure. Default is empty. 
- `show`: *optional*. Displays the result. Default is `True`. 

## `te_temperament_measures.py`

Analyses tunings and computes temperament measures from the temperament mapping matrix. 

Requires `te_common`, `te_optimizer`, and optionally `te_symbolic`. 

Use `Temperament` to construct a temperament object. Methods: 
- `tune`: calls `wrapper_main`/`wrapper_symbolic` and shows the generator, tuning map, error map, tuning error, and tuning bias. Parameters: 
	- `optimizer`: *optional*. Optimizer. `"main"`: calls `wrapper_main`. `"sym"`: calls `wrapper_symbolic`. Default is `"main"`. 
	- `norm`: *optional*. Norm profile for the tuning space. See above. 
	- `inharmonic`: *optional*. For subgroup temperaments, treats the basis as if they were primes. Default is `False`. 
	- `constraint`: *optional*. Constrains this subgroup to pure. Default is empty. 
	- `destretch`: *optional*. Destretches this ratio to pure. Default is empty. 
- `temperament_measures`: shows the complexity, error, and badness (simple and logflat). Parameters: 
	- `ntype`: *optional*. Averaging normalizer. Has `"breed"` (default), `"smith"` and `"none"`. 
	- `norm`: *optional*. Norm profile for the tuning space. See above. 
	- `error_scale`: *optional*. Scales the error. Default is `1200` (cents).
	- `badness_scale`: *optional*. Scales the badness. Default is `1000` (millioctaves). 
- `wedgie`: returns and shows the wedgie of the temperament. 
- `comma_basis`: returns and shows the comma basis of the temperament. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as an array of column vectors.** 

## `te_equal.py`

Tools related to equal temperaments. 
- Constructs higher-rank temperaments using equal temperaments. 
- Finds the GPVs from the comma list. 

Requires `te_common`, `te_optimizer`, and `te_temperament_measures`. 

Use `et_construct` to quickly construct temperaments from equal temperaments. Parameters: 
- `et_list`: *first positional*, *required*. The equal temperament list. 
- `subgroup`: *second positional*, *required*. The subgroup for the equal temperament list. 

Use `et_sequence` to iterate through all GPVs. Parameters: 
- `monzos`: *optional\**. Specifies the commas to be tempered out. Default is empty, implying **JI**. 
- `subgroup`: *optional\**. Custom subgroup for the map. Default is prime harmonics. 
	- \* At least one of the above must be specified, for the script to know the dimension. 
- `ntype`: *optional*. Averaging normalizer. See above. 
- `norm`: *optional*. Norm profile for the tuning space. See above. 
- `inharmonic`: *optional*. For subgroup temperaments, treats the basis as if they were primes. Default is `False`. 
- `cond`: *optional*. Either `"error"` or `"badness"`. Default is `"error"`. 
- `pv`: *optional*. If `True`, only patent vals will be considered. Default is `False`. 
- `prog`: *optional*. If `True`, threshold will be updated. Default is `True`. 
- `threshold`: *optional*. Temperaments failing this will not be shown. Default is `20`. 
- `search_range`: *optional*. Specifies the upper bound where to stop searching. Default is `1200`. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as an array of column vectors.** 

## `te_lattice.py`

Not fully functional yet. Currently able to find the complexity spectrum from the temperament map. 

Requires `te_common` and `te_temperament_measures`. 

Use `TemperamentLattice` to construct a temperament object. Methods: 
- `find_temperamental_norm`: shows the temperamental complexity of an interval. 
- `find_complexity_spectrum`: shows the complexity spectrum of a temperament
