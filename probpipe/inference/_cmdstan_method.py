"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

from ._delegating_method import make_delegating_method


CmdStanNutsMethod = make_delegating_method(
    method_name="cmdstan_nuts",
    model_path="probpipe.modeling._stan.StanModel",
    method_priority=70,
)
