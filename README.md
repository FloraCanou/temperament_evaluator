# Temperament Evaluator

## Dependencies

- [SciPy](https://scipy.org/)
	- various tasks including optimization, temperament measures, wedgie, etc. 
- [SymPy](https://www.sympy.org/en/index.html)
	- various tasks including symbolic solution, mapping normalization, comma basis, etc. 

## `te_common.py`

Common functions. Required by virtually all subsequent modules. 

Use the `Norm` class to create a norm profile for the tuning space. Parameters: 
- `wtype`: Weight method. Has `"tenney"` (default), `"equilateral"`, and `"wilson"`/`"benedetti"`. 
- `wamount`: Weight scaling factor. Default is `1`. 
- `skew`: Skew. This is Mike Battaglia's *k*. Default is `0`, meaning no skew. For **Weil**, use `1`. 
- `order`: Order. Default is `2`, meaning **Euclidean**. For **XOP tuning**, use `np.inf`. 

## `te_optimizer.py`

Optimizes tunings. Custom norm profile, constraints and destretch are supported. *It is recommended to use `te_temperament_measures` instead since it calls this module and it has a more accessible interface.*

Requires `te_common`. 

Use `optimizer_main` to optimize a temperament. Parameters: 
- `vals`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Specifies a custom subgroup for the map. Default is prime harmonics. 
- `norm`: *optional*. Specifies the norm profile for the tuning space. See above. 
- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. Default is empty. 
- `des_monzo`: *optional*. Destretches this monzo to pure. Default is empty. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as composed by column vectors.** 

## `te_optimizer_legacy.py`

Legacy single-file edition (doesn't require `te_common.py`). 

## `te_symbolic.py`

Solves Euclidean tunings symbolically. *It is recommended to use `te_temperament_measures` instead since it calls this module and it has a more accessible interface.*

Requires `te_common`. 

Use `symbolic` to solve for a Euclidean tuning of a temperament. Parameters: 
- `vals`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Specifies a custom subgroup for the map. Default is prime harmonics. 
- `norm`: *optional*. Specifies the norm profile for the tuning space. See above. 
- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. Default is empty. 
- `des_monzo`: *optional*. Destretches this monzo to pure. Default is empty. 

## `te_temperament_measures.py`

Analyses tunings and computes temperament measures from the temperament map. 

Requires `te_common`, `te_optimizer`, and optionally `te_symbolic`. 

Use `Temperament` to construct a temperament object. Methods: 
- `tune`: calls `optimizer_main`/`symbolic` and shows the generator, tuning map, mistuning map, tuning error, and tuning bias. Parameters: 
	- `optimizer`: *optional*. Specifies the optimizer. `"main"`: calls `optimizer_main`. `"sym"`: calls `symbolic`. Default is `"main"`. 
	- `norm`: *optional*. Specifies the norm profile for the tuning space. See above. 
	- `enforce`: *optional*. A shortcut to specify constraints and destretch targets, so you don't need to enter monzos. Default is empty. To add an enforcement, use `c` or `d` followed by the subgroup index. For example, if the subgroup is the prime harmonics: 
		- `"c"` or `"c1"`: pure-2 constrained
		- `"d"` or `"d1"`: pure-2 destretched
		- `"c1c2"`: pure-2.3 constrained
		- `"c0"`: a special indicator meaning weight-skewed-ones constrained
	- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. Default is empty. Overrides `enforce`. 
	- `des_monzo`: *optional*. Destretches this monzo to pure. Default is empty. Overrides `enforce`. 
- `temperament_measures`: shows the complexity, error, and badness (simple and logflat). Parameters: 
	- `ntype`: *optional*. Specifies the averaging normalizer. Has `"breed"` (default), `"smith"` and `"none"`. 
	- `norm`: *optional*. Specifies the norm profile for the tuning space. See above. 
	- `badness_scale`: *optional*. Scales the badness, literally. Default is `1000`. 
- `wedgie`: returns and shows the wedgie of the temperament. 
- `comma_basis`: returns and shows the comma basis of the temperament. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as composed by column vectors.** 

## `te_equal.py`

Tools related to equal temperaments. 
- Constructs higher-rank temperaments using equal temperaments. 
- Finds the GPVs from the comma list. 

Requires `te_common`, `te_optimizer`, and `te_temperament_measures`. 

Use `et_construct` to quickly construct temperaments from equal temperaments. Parameters: 
- `et_list`: *first positional*, *required*. The equal temperament list. 
- `subgroup`: *second positional*, *required*. The subgroup for the equal temperament list. 
- `alt_val`: *optional*. Alters the mapping by this matrix. 

Use `et_sequence` to iterate through all GPVs. Parameters: 
- `monzo_list`: *optional\**. Specifies the commas to be tempered out. Default is empty, implying **JI**. 
- `subgroup`: *optional\**. Specifies a custom subgroup for the map. Default is prime harmonics. 
	- \* At least one of the above must be specified, for the script to know the dimension. 
- `cond`: *optional*. Either `"error"` or `"badness"`. Default is `"error"`. 
- `ntype`: *optional*. Specifies the averaging normalizer. See above. 
- `norm`: *optional*. Specifies the norm profile for the tuning space. See above. 
- `pv`: *optional*. If `True`, only patent vals will be considered. Default is `False`. 
- `prog`: *optional*. If `True`, threshold will be updated. Default is `True`. 
- `threshold`: *optional*. Temperaments failing this will not be shown. Default is `20`. 
- `search_range`: *optional*. Specifies the upper bound where to stop searching. Default is `1200`. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as composed by column vectors.** 

## `te_lattice.py`

Not fully functional yet. Currently able to find the complexity spectrum from the temperament map. 

Requires `te_common` and `te_temperament_measures`. 

Use `TemperamentLattice` to construct a temperament object. Methods: 
- `find_temperamental_norm`: shows the temperamental complexity of an interval. 
- `find_complexity_spectrum`: shows the odd-limit complexity spectrum of a temperament
