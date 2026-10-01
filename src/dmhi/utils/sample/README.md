# Synthetic smoke sample

`X.npy` / `M.npy` are a small, fully synthetic multivariate time-series sample
(`40` windows x `24` steps x `5` channels), generated deterministically by
`../../src/dmhi/utils/make_sample.py`. They exist only to let `tests/test_smoke.py` run the real
imputation pipeline end to end in seconds without any data download. They are not
used for any reported result. Regenerate with:

```bash
python ../../src/dmhi/utils/make_sample.py
```
