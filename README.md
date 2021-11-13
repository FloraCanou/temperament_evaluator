# Tuning Optimizer & TE Temperament Measures Calculator

## `tuning_optimizer.py`

Optimizes tunings. Custom order, constraints and stretches are supported. 

Use `optimizer_main` to optimize, yet *the next module calls this function with presets so it is recommended to use that instead.* Parameters: 
- `map`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Specifies a custom subgroup for the map. Default is prime harmonics. 
- `order`: *optional*. Specifies the order of the norm to be minimized. Default is `2`, meaning **TE**. For **TOP**, use `np.inf`. 
- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. Default is empty. 
- `stretch_monzo`: *optional*. Stretches this monzo to pure. Default is empty. 

**Important: monzos must be entered as column vectors**. 

## `te_temperament_measures.py`

Analyses tunings and computes TE temperament measures from the temperament map. Requires the above module. 

Use `Temperament` to construct a temperament object. Methods: 
- `analyse`: calls `optimizer_main` and shows the generator, tuning map, mistuning map, tuning error, and tuning bias. Parameters: 
	- `type`: *optional*. Has presets `"te"`, `"pote"`, `"cte"`, `"top"`, `"potop"`, `"ctop"` and `"custom"` (default). Only in `"custom"` can you specify your own constraints and stretch goals. 
	- `order`: *optional*, *only works if type is "custom"*. Specifies the order of the norm to be minimized. Default is `2`, meaning **TE**. For **TOP**, use `np.inf`. 
	- `cons_monzo_list`: *optional*, *only works if type is "custom"*. Constrains this list of monzos to pure. Default is empty. 
	- `stretch_monzo`: *optional*, *only works if type is "custom"*. Stretches this monzo to pure. Default is empty. 
- `temperament_measures`: shows the TE complexity, TE error, TE badness (simple and logflat). Parameters: 
	- `type`: *optional*. Has `"rmsgraham"` (default), `"rmsgene"` and `"l2"`. 
	- `badness_scale`: *optional*. Scales the badness, literally. Default is `100`. 

**Important: monzos must be entered as column vectors**. 

## `et_sequence_error.py`

Finds the ET sequence from the comma list. Can be used to find optimal patent vals. Requires the above module. 

Use `et_construct` to quickly construct an equal temperament. Parameters: 
- `n`: *first positional*, *required*. The equal temperament number. 
- `subgroup`: *second positional*, *required*. The subgroup for the equal temperament. 
- `alt_val`: *optional*. Alters the val by this row vector. 

Use `et_sequence_error` to iterate through all GPVs. Parameters: 
- `monzo_list`: *optional\**. Specifies the commas to be tempered out. Default is empty, implying **JI**. 
- `subgroup`: *optional\**. Specifies a custom subgroup for the map. Default is prime harmonics. 
	- \* At least one of the above must be specified, for the script to know the dimension. 
- `cond`: *optional*. Either `"error"` or `"badness"`. Default is `"error"`. 
- `threshold`: *optional*. Temperaments failing this will not be shown. 
- `prog`: *optional*. If `True`, threshold will be updated. Default is `True`. 
- `pv`: *optional*. If `True`, only patent vals will be considered. Default is `False`. 
- `search_range`: *optional*. Specifies the upper bound where to stop searching. Default is `1200`. 

**Important: monzos must be entered as column vectors**. 

## `complexity_spectrum.py`

Finds the complexity spectrum from the temperament map. 

See `examples.py`. 
