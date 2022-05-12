# Temperament Evaluator

## Dependencies

- [SciPy](https://scipy.org/)
- [SymPy](https://www.sympy.org/en/index.html)

## `te_common.py`

Common functions. Required by virtually all subsequent modules. 

## `te_optimizer.py`

Optimizes tunings. Custom norm order, constraints and stretches are supported. 

Requires `te_common`. 

Use `optimizer_main` to optimize, yet *it is recommended to use the next module instead since it calls this function with presets.* Parameters: 
- `map`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Specifies a custom subgroup for the map. Default is prime harmonics. 
- `wtype`: *optional*. Specifies the weighter. Has `"tenney"` (default), `"frobenius"`, `"inverse tenney"`, `"benedetti"` and `"weil"`. 
- `order`: *optional*. Specifies the order of the norm to be minimized. Default is `2`, meaning **Euclidean**. For **TOP tuning**, use `np.inf`. 
- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. Default is empty. 
- `des_monzo`: *optional*. Destretches this monzo to pure. Default is empty. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as composed by column vectors.** 

## `te_optimizer_legacy.py`
Legacy single-file edition (doesn't require `te_common.py`). 

## `te_temperament_measures.py`

Analyses tunings and computes temperament measures from the temperament map. 

Requires `te_common` and `te_optimizer`. 

Use `Temperament` to construct a temperament object. Methods: 
- `analyse`: calls `optimizer_main` and shows the generator, tuning map, mistuning map, tuning error, and tuning bias. Parameters: 
	- `wtype`: *optional*, *only works if type is "custom"*. Specifies the weighter. See above. 
	- `order`: *optional*, *only works if type is "custom"*. Specifies the order of the norm to be minimized. See above. 
	- `enforce`: *optional*. Has  `"d"`, `"c"`, `"xoc"` and `"custom"` (default). Only in `"custom"` can you specify your own constraints and destretch goals. 
		- `"d"`: pure-octave destretched
		- `"c"`: pure-octave constrained
		- `"xoc"`: \[weighter type\]-ones constrained
		- `"none"`: no enforcement (disregard the following parameters)
	- `cons_monzo_list`: *optional*, *only works if type is "custom"*. Constrains this list of monzos to pure. Default is empty. 
	- `des_monzo`: *optional*, *only works if type is "custom"*. Destretches this monzo to pure. Default is empty. 
- `temperament_measures`: shows the complexity, error, and badness (simple and logflat). Parameters: 
	- `ntype`: *optional*. Specifies the averaging method. Has `"breed"` (default), `"smith"` and `"l2"`. 
	- `wtype`: *optional*. Specifies the weighter. See above. 
	- `badness_scale`: *optional*. Scales the badness, literally. Default is `100`. 
- `comma_basis`: returns and shows the comma basis of the temperament. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as composed by column vectors.** 

## `te_equal.py`

Tools related to equal temperaments. Constructs higher-rank temperaments using equal temperaments. Finds the GPVs from the comma list. 

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
- `ntype`: *optional*. Specifies the averaging method. See above. 
- `wtype`: *optional*. Specifies the weighter. See above. 
- `pv`: *optional*. If `True`, only patent vals will be considered. Default is `False`. 
- `prog`: *optional*. If `True`, threshold will be updated. Default is `True`. 
- `threshold`: *optional*. Temperaments failing this will not be shown. Default is `20`. 
- `search_range`: *optional*. Specifies the upper bound where to stop searching. Default is `1200`. 

**Important: a single monzo should be entered as a vector. A monzo list should be entered as composed by column vectors.** 

## `te_lattice.py`

Not fully functional yet. Currently able to find the complexity spectrum from the temperament map. 

Requires `te_common`. 
