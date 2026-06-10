def test_diagnostics_public_api_imports():
    from probpipe.diagnostics import DiagnosticsModule, run_ppc, loo

    assert DiagnosticsModule is not None
    assert run_ppc is not None
    assert loo is not None