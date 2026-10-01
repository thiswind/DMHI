# Contributing

Thanks for your interest. This repository accompanies a paper under review;
development is currently focused on correctness and reproducibility.

- Dev install: `pip install -e ".[dev]"`
- Before opening a PR, make sure both of these pass:
  `python tests/test_smoke.py` and `python -m pytest tests -q`
- Keep the core package (`dmhi`) dependency-light; baseline-only dependencies
  belong in the `baselines` extra.
- Issues with a minimal reproducer are the fastest way to get a fix.
- All contributions are licensed under the repository license (MIT).
