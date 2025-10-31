# custom_types.py
"""
These type definitions and aliases are intended to eventually be generalized
to support both numpy and jax array backends. 

We generally following the conventions:  
- Annotate function input with `ArrayLike`  
- Annotate function output with `Array`
"""

from typing import TypeAlias, TypeVar
from numpy.random import Generator as NumpyRNG

from numpy.typing import (
    NDArray as NumpyArray, 
    ArrayLike as NumpyArrayLike
)

from numpy import (
    floating as NumpyFloating,
    number as NumpyNumber
)

T = TypeVar("T")

Float: TypeAlias = NumpyFloating
Array: TypeAlias = NumpyArray
ArrayLike: TypeAlias = NumpyArrayLike
PRNG: TypeAlias = NumpyRNG