"""Joint distributions built on ValuesDistribution.

Re-export facade --- all concrete classes live in their own modules:

*  :mod:`._product`           --- :class:`ProductDistribution`
*  :mod:`._sequential_joint`  --- :class:`SequentialJointDistribution`
*  :mod:`._joint_empirical`   --- :class:`JointEmpirical`
*  :mod:`._joint_gaussian`    --- :class:`JointGaussian`
"""

from ._product import ProductDistribution
from ._sequential_joint import SequentialJointDistribution
from ._joint_empirical import JointEmpirical
from ._joint_gaussian import JointGaussian

__all__ = [
    "ProductDistribution",
    "SequentialJointDistribution",
    "JointEmpirical",
    "JointGaussian",
]
