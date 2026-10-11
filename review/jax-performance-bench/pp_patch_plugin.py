"""A pytest plugin that applies patches.py in the mode PP_PATCH names, when it is set.

Usage: PP_PATCH=kernel PYTHONPATH=review/jax-performance-bench uv run pytest -p pp_patch_plugin ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = os.environ.get("PP_PATCH")
if mode:
    import patches

    patches.apply(mode)
