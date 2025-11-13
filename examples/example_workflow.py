from typing import Optional, Callable
from probpipe import WorkFlow, Module 
from prefect import flow, task
import numpy as np
from probpipe.core.multivariate import Normal1D
from numpy.typing import NDArray

wf = WorkFlow()

@wf.run_func(as_task=True)
def square(x: int) -> int:
    return x * x

mod = square()
print(mod.square(5))  # prints 25