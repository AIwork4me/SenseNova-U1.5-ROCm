"""Pytest collection rules for the offline unit-test gate.

CONTRIBUTING.md requires `python3 -m pytest tests/ -q` to pass offline,
without a GPU. tests/validation/test_pytorch_nightly_bug*.py violate both
premises by design: they are standalone GPU repro launchers (module-level
torch code ending in `sys.exit(0)`) meant to be executed one-per-process by
tests/validation/run_all_nightly.sh — collecting them under pytest aborts
with INTERNALERROR (SystemExit at import) and would touch the GPU from the
offline unit gate. Exclude exactly those; everything else under tests/
stays collectable.
"""
collect_ignore_glob = ["validation/test_pytorch_nightly_bug*.py"]
