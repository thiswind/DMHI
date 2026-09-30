Analysis-runner provenance (ALLOW pull, 2026-08-02)
===================================================

Policy: gap_provenance_audit plan section 6.1, ALLOW-only
        (effective runners + MD k-sensitivity analysis)

Pulled into code/experiments/:
  md_k_sensitivity.py
  downstream_md_auroc.py
  downstream_gait_classify.py
  determinism_check.py
  downstream_gas_classify.py
  _code_root.py          (local path adapter)

Pulled into paper/assets/data_derived/:
  md_ksens.jsonl         (36 records, k in {3,5,10,20})

Locked (not overwritten, paper SSOT unchanged):
  downstream_physionet_summary.csv
  downstream_gait_summary.csv
  downstream_gas_summary.csv

Not pulled (DENY / HOLD):
  e4d/e4f summary CSVs (stale vs paper SSOT)
  ablation2|4 use_decoder logs
  stage3_auto_loop / stage3_rewrite results
  exp5_ksens_multidomain.json (HOLD FAIL 2026-08-02)

Run notes:
  These analysis runners expect the full server-side data tree
  (data/processed/ plus experiments/logs/* with checkpoints and
  baseline imputations); they are provenance/re-run tools, not a
  laptop-only smoke path. They resolve the repo root automatically
  via _code_root.py, e.g.:
      python experiments/downstream_md_auroc.py
