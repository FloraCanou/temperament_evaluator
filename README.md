# Tuning Optimizer & TE Temperament Measures Calculator

## `tuning_optimizer.py`

Optimizes tunings. Custom order, constraints and stretches are supported. 

Use `optimizer_main` to optimize, yet *the next module calls this function with presets so it is recommended to use that instead.* Parameters: 
- `map`: *first positional*, *required*. The map of the temperament. 
- `subgroup`: *optional*. Specifies a custom subgroup for the map. Default is the primes. 
- `order`: *optional*. Specifies the order of the norm to be minimized. Default is `2`, meaning **TE**. For **TOP**, use `np.inf`. 
- `cons_monzo_list`: *optional*. Constrains this list of monzos to pure. **Monzos must be entered as column vectors**. Default is empty. 
- `stretch_monzo`: *optional*. Stretches this monzo to pure. **Monzos must be entered as column vectors**. Default is empty. 

## `te_temperament_measures.py`

Analyses tunings and computes TE temperament measures from the temperament map. Requires the above module. 

Use `Temperament` to construct a temperament object. Methods: 
- `analyse`: calls `optimizer_main` and shows the generator, tuning map, mistuning map, tuning error, and tuning bias. Parameters: 
	- `type`: *optional*. Has presets `"te"`, `"pote"`, `"cte"`, `"top"`, `"potop"`, `"ctop"` and `"custom"` (default). Only in `"custom"` can you specify your own constraints and stretch goals. 
	- `order`: *optional*, *only works if type is "custom"*. Specifies the order of the norm to be minimized. Default is `2`, meaning **TE**. For **TOP**, use `np.inf`. 
	- `cons_monzo_list`: *optional*, *only works if type is "custom"*. Constrains this list of monzos to pure. **Monzos must be entered as column vectors**. Default is empty. 
	- `stretch_monzo`: *optional*, *only works if type is "custom"*. Stretches this monzo to pure. **Monzos must be entered as column vectors**. Default is empty. 
- `temperament_measures`: shows the TE complexity, TE error, TE badness (simple and logflat). Parameters: 
	- `type`: *optional*. Has `"rmsgraham"` (default), `"rmsgene"` and `"l2"`. 
	- `badness_scale`: *optional*. Scales the badness, literally. Default is `100`. 

## `et_sequence_error.py`

Finds the ET sequence from the comma list. Can be used to find optimal patent vals. Requires the above module. 

Use `et_construct` to construct an equal temperament. 

Use `et_sequence_error` to iterate through all GPVs. 

## `complexity_spectrum.py`

Finds the complexity spectrum from the temperament map. 

See `examples.py` for more documents. 
