from typing import Optional, Callable
from probpipe import Module, Workflow
from prefect import flow, task
import numpy as np
from numpy.typing import NDArray
from probpipe import Normal1D, Distribution, DistributionModule
from types import MappingProxyType
import inspect


class TransportMap:
    def map(self, dist: Distribution) -> Distribution:
        raise NotImplementedError

class AffineTransform(TransportMap):
    def map(self, dist: Distribution) -> Distribution:
        print(f"Transforming: {dist}")
        return dist  # Example: identity


def my_workflow(prior: Distribution, tmap: TransportMap) -> Distribution:
    return tmap.map(prior)


MyModule = Workflow.create(my_workflow)  # Prefect task

aff_map = AffineTransform()
mod = MyModule(tmap=aff_map)

prior_dist = Normal1D(mu=0, sigma=1)
out = mod.run(prior=prior_dist, tmap=aff_map)  


