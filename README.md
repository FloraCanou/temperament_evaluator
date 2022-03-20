# Tuning Optimizer & TE Temperament Measures Calculator

## Dependencies

- [SciPy](https://scipy.org/)
- [SymPy](https://www.sympy.org/en/index.html)

## `tuning_optimizer.py`

Optimizes tunings. Custom norm order, constraints and stretches are supported. 

Use `optimizer_main` to optimize, yet *it is recommended to use the next module instead since it calls this function with presets.* Parameters: 
- `map`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Specifies a custom subgroup for the map. Default is prime harmonics. 
- `wtype`: *optional*. Specifies the weighter. Has `"tenney"` (default), `"frobenius"`, and `"partch"`. 
- `order`: *optional*. Specifies the order of the norm to be minimized. Default is `2`, meaning **Euclidean**. For **TOP tuning**, use `np.inf`. 
- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. Default is empty. 
- `stretch_monzo`: *optional*. Stretches this monzo to pure. Default is empty. 

**Important: monzos must be entered as column vectors**. 

## `te_temperament_measures.py`

Analyses tunings and computes TE temperament measures from the temperament map. Requires the above module. 

Use `Temperament` to construct a temperament object. Methods: 
- `analyse`: calls `optimizer_main` and shows the generator, tuning map, mistuning map, tuning error, and tuning bias. Parameters: 
	- `wtype`: *optional*, *only works if type is "custom"*. Specifies the weighter. Has `"tenney"` (default), `"frobenius"`, and `"partch"`. 
	- `order`: *optional*, *only works if type is "custom"*. Specifies the order of the norm to be minimized. Default is `2`, meaning **Euclidean**. For **TOP tuning**, use `np.inf`. 
	- `enforce`: *optional*. Has  `"po"`, `"c"`, `"xoc"` and `"custom"` (default). Only in `"custom"` can you specify your own constraints and stretch goals. 
		- `"po"`: pure-octave stretched
		- `"c"`: pure-octave constrained
		- `"xoc"`: \[weighter type\]-ones constrained
		- `"none"`: no enforcement (disregard the following parameters)
	- `cons_monzo_list`: *optional*, *only works if type is "custom"*. Constrains this list of monzos to pure. Default is empty. 
	- `stretch_monzo`: *optional*, *only works if type is "custom"*. Stretches this monzo to pure. Default is empty. 
- `temperament_measures`: shows the complexity, error, and badness (simple and logflat). Parameters: 
	- `ntype`: *optional*. Specifies the averaging method. Has `"breed"` (default), `"smith"` and `"l2"`. 
	- `wtype`: *optional*. Specifies the weighter. Has `"tenney"` (default), `"frobenius"`, and `"partch"`. 
	- `badness_scale`: *optional*. Scales the badness, literally. Default is `100`. 

**Important: monzos must be entered as column vectors**. 

## `et_sequence_error.py`

Finds the ET sequence from the comma list. Can be used to find optimal patent vals. Requires the above module. 

Use `et_construct` to quickly construct temperaments from equal temperaments. Parameters: 
- `et_list`: *first positional*, *required*. The equal temperament list. 
- `subgroup`: *second positional*, *required*. The subgroup for the equal temperament list. 
- `alt_val`: *optional*. Alters the mapping by this matrix. 

Use `et_sequence_error` to iterate through all GPVs. Parameters: 
- `monzo_list`: *optional\**. Specifies the commas to be tempered out. Default is empty, implying **JI**. 
- `subgroup`: *optional\**. Specifies a custom subgroup for the map. Default is prime harmonics. 
	- \* At least one of the above must be specified, for the script to know the dimension. 
- `cond`: *optional*. Either `"error"` or `"badness"`. Default is `"error"`. 
- `ntype`: *optional*. Specifies the averaging method. Has `"breed"` (default), `"smith"` and `"l2"`. 
- `wtype`: *optional*. Specifies the weighter. Has `"tenney"` (default), `"frobenius"`, and `"partch"`. 
- `pv`: *optional*. If `True`, only patent vals will be considered. Default is `False`. 
- `prog`: *optional*. If `True`, threshold will be updated. Default is `True`. 
- `threshold`: *optional*. Temperaments failing this will not be shown. Default is `20`. 
- `search_range`: *optional*. Specifies the upper bound where to stop searching. Default is `1200`. 

**Important: monzos must be entered as column vectors**. 

## `complexity_spectrum.py`

Finds the complexity spectrum from the temperament map. 

See `examples.py`. 
