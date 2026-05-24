# Temperament Evaluator

This Python 3 script can be used to compute various regular temperament data.

## Requirements

- [SciPy](https://scipy.org/)
	- various tasks including optimization, temperament measures, wedgie, etc. 
- [SymPy](https://www.sympy.org/en/index.html)
	- various tasks including symbolic solution, mapping normalization, comma basis, etc. 
- [tqdm](https://tqdm.github.io/)
	- to render the progress bar for `te_equal.et_sequence`. 

```
pip install scipy sympy tqdm
```

## Usage Guide

See our [wiki](https://github.com/FloraCanou/temperament_evaluator/wiki) for technical references. 

### Setup

```
import numpy as np
from lib.te_common import *
from lib.te_temperament_measures import *
from lib.te_equal import *
from lib.te_lattice import *
```

Set precision level ([note](https://github.com/FloraCanou/temperament_evaluator/wiki/Precision-limits))

```
np.set_printoptions (precision = 3)
```

### Basic usages

To construct a temperament

```
# magic
temp = Temperament ([[1, 0, 2, -1], [0, 5, 1, 12]]) 
```

To tune the temperament

```
# cte tuning
temp.tune (constraint = Subgroup ([2])) 
```
```
# cwe tuning
temp.tune (norm = Norm (skew = 1), constraint = Subgroup ([2])) 
```
```
# pote tuning
temp.tune (destretch = Ratio (2, 1)) 
```

To find the temperament measures

```
temp.temperament_measures (ntype = "sintel") 
```

To find the wedgie for the temperament

```
temp.wedgie ()
```

To find a comma basis for the temperament
```
temp.comma_basis ()
```

To find the optimal GPV sequence for the temperament

```
et_sequence (
    temp.comma_basis (show = False), cond = "error", search_range = 300)
```

### Alternative ways to construct temperaments

To construct a temperament from equal temperaments

```
# squares
temp = et_construct (["14c", "17c"], Subgroup ([2, 3, 5, 7]))
```
```
# bps
temp = et_construct (["b4", "b13"], Subgroup ([3, 5, 7]))
```

To construct a temperament from a comma basis

```
# squares
temp = Temperament.from_comma_space (
    Subgroup (["81/80", "2401/2400"]))
```
```
# bps
temp = Temperament.from_comma_space (
    Subgroup (["245/243"]), Subgroup ([3, 5, 7]))
```

### Lattice-related functions

This is mainly used to find the octave-equivalent interval temperamental complexity spectrum. To do this, we need to construct the temperament with `te_lattice.TemperamentLattice`. Here we're demonstrating using tridecimal history. 

```
# tridecimal history
temp = TemperamentLattice (
    [[1, 2, 0, 0, 1, 2], 
    [0, 6, 0, -7, -2, 9], 
    [0, 0, 1, 1, 1, 1]]
)
temp.temperamental_complexity_spectrum (
    diamond_monzos_gen (15, eq = 2, comp = False), oe = True)
```
