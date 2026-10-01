Analysis-runner provenance
==========================

Policy: gap_provenance_audit plan section 6.1, ALLOW-only
        (effective runners + MD k-sensitivity analysis)

Provenance runners in `scripts/`:
  md_k_sensitivity.py
  downstream_md_auroc.py
  downstream_gait_classify.py
  determinism_check.py
  downstream_gas_classify.py

Derived asset archived in `paper/assets/data_derived/`:
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
  These analysis runners expect the full data tree (data/processed/
  plus runs/* holding checkpoints and baseline imputations); they are
  provenance/re-run tools, not a laptop-only smoke path. They resolve
  the repository root automatically from their own location, e.g.:

      python scripts/downstream_md_auroc.py
      python scripts/md_k_sensitivity.py --out runs/md_k_sensitivity/md_ksens.jsonl
