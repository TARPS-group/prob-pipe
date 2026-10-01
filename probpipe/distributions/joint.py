"""Joint distributions built on RecordDistribution.

Re-export facade --- all concrete classes live in their own modules:

*  :mod:`._product`           --- :class:`ProductDistribution`
*  :mod:`._sequential_joint`  --- :class:`SequentialJointDistribution`
*  :mod:`._joint_gaussian`    --- :class:`JointGaussian`
"""

from ._joint_gaussian import JointGaussian
from ._product import ProductDistribution
from ._sequential_joint import SequentialJointDistribution

__all__ = [
    "JointGaussian",
    "ProductDistribution",
    "SequentialJointDistribution",
]
