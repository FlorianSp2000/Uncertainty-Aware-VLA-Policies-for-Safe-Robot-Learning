# From Octo (https://github.com/octo-models/octo, MIT) via the V-GPS fork (https://github.com/nakamotoo/octo). Modified for this thesis.
from typing import Any, Mapping, Sequence, Union

import jax

PRNGKey = getattr(jax.random, "KeyArray", jax.Array)  # KeyArray removed in JAX 0.5
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
