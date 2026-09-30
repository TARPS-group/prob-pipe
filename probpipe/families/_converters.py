"""The shipped converters between the catalog's families and backend representations.

A converter moves a law to another representation, preserving its event
declaration, at a recorded fidelity. The shipped converters cover the
parametric families, their backend distributions, and the moment-matched and
sample-based stand-ins.

The converters are defined in :mod:`probpipe.converters`, which registers them
with the global converter registry at import.
"""

from __future__ import annotations

__all__: list[str] = []
