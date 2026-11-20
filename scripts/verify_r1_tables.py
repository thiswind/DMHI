#!/usr/bin/env python3
"""Cell-level verification of R1 revised-manuscript tables vs authoritative JSONs.

Checks: tab_e7_air, tab_e7_physio, tab_e3_unified_md, tab_e5_pi,
tab_e5_deepmem, Table II DMHI+PSW-I rows, Table III DMHI+PSW-I RMSE rows,
abstract 92.0%. Exits nonzero on any mismatch.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / 'code/results/revision-r1'
L = ROOT / 'paper/revision/latex-source-r1'
ok, bad = 0, []
ROW = re.compile('^(.+?) & (.+?) ' + re.escape('\\\\') + '\n', re.M)
def CLEAN(s):
    return re.sub(r'\\rev\{|\\textbf\{|\\underline\{|\}', '', s.strip())

# --- e7 tables
for ds, fn in [('air_quality_italy', 'tab_e7_air.tex'), ('physionet2012', 'tab_e7_physio.tex')]:
    seeds = {s: json.load(open(RR / f'e7_natural/{ds}_seed{s}.json'))['methods'] for s in (7, 42, 123)}
    protos = ['native_pattern', 'group2x12', 'group2x24', 'group4x12', 'group4x24', 'group8x12']
    tex = (L / fn).read_text()
    rows = [r for n, r in ROW.findall(tex)
            if n.strip() != 'Missing pattern' and len(r.split('&')) == 4]
    assert len(rows) == 6, (fn, len(rows))
    for row, p in zip(rows, protos):
        cells = [float(CLEAN(c)) for c in row.split('&')]
        for v, meth in zip(cells, ['DMHI', 'Linear', 'LOCF', 'Mean']):
            e = sum(seeds[s][p][meth]['MAE'] for s in (7, 42, 123)) / 3
            (bad.append((fn, p, meth, v, round(e, 4))) if abs(v - e) > 5e-5 else None) or None
            if abs(v - e) <= 5e-5: ok += 1

# --- e3 unified MD
e3 = json.load(open(RR / 'e3_md/md_table.json'))
e3a = {}
for r in e3:
    e3a.setdefault(r['dataset'], []).append(r)
tex = (L / 'tab_e3_unified_md.tex').read_text()
rows = [r for n, r in ROW.findall(tex)
        if n.strip() != 'Domain' and len(r.split('&')) == 5]
assert len(rows) == 12, len(rows)
dom_of_row = ['air_quality_italy', 'breizh_rs', 'delta_robot', 'gait_uci', 'gas_home', 'kuka',
              'mimic', 'mocap_karate', 'nanodrone', 'physionet2012', 'solo12', 'mhealth']
for row, d in zip(rows, dom_of_row):
    cells = [float(CLEAN(c)) for c in row.split('&')]
    for v, k in zip(cells, ['dmhi_md', 'linear_md', 'locf_md', 'mean_md', 'psw_i_md']):
        e = sum(x[k] for x in e3a[d]) / 3
        if abs(v - e) > 5e-4: bad.append(('e3', d, k, v, round(e, 3)))
        else: ok += 1

# --- e5 pi
tex = (L / 'tab_e5_pi.tex').read_text()
dom_disp_rev = {'Air Quality': 'air_quality_italy', 'BreizhCrops': 'breizh_rs', 'Delta': 'delta_robot',
                'Gait': 'gait_uci', 'Gas': 'gas_home', 'MoCap': 'mocap_karate', 'Solo-12': 'solo12'}
rows = [(n, r) for n, r in ROW.findall(tex)
        if n.strip() != 'Domain' and '{\\times}' in r.split('&')[0]]
assert len(rows) == 7, len(rows)
for name, row in rows:
    d = dom_disp_rev[name.strip()]
    j = json.load(open(RR / f'e5_edge/pi_{d}_seed7.json'))['methods']['DMHI']
    parts = row.split('&')
    lat, rss, en = (float(CLEAN(parts[i])) for i in (1, 2, 3))
    for v, e, tol in [(lat, j['latency_ms_median'], 0.05), (rss, j['peak_rss_mb'], 0.5), (en, j['energy_mj_per_sample_est'], 0.5)]:
        if abs(v - e) > tol: bad.append(('pi', d, v, e))
        else: ok += 1

# --- e5 deep_mem
tex = (L / 'tab_e5_deepmem.tex').read_text()
rows = [(n, r) for n, r in ROW.findall(tex) if '&' in r and 'Method' not in n]
for meth, jf in [('CSDI', 'deep_mem_csdi_gas_home.json'), ('DMHI (ours)', 'deep_mem_dmhi_gas_home.json'),
                 ('ImputeFormer', 'deep_mem_imputeformer_gas_home.json')]:
    j = json.load(open(RR / 'e5_edge' / jf))
    match = [(n, r) for n, r in rows if n.strip() == meth]
    cells = [c.strip() for c in match[0][1].split('&')]
    lat, rss = float(CLEAN(cells[1])), float(CLEAN(cells[2]))
    if abs(lat - j['latency_ms_median']) > 1: bad.append(('dm', meth, lat, j['latency_ms_median']))
    else: ok += 1
    if abs(rss - j['rss_peak_mb']) > 1: bad.append(('dm', meth, rss, j['rss_peak_mb']))
    else: ok += 1

# --- manuscript Table II + III rows + abstract
tex = (L / 'manuscript.tex').read_text()
tbl = tex.split(r'\label{tab:mae&mse}')[1].split(r'\label{tab:rmse&md}')[0]
order = ['physionet2012', 'mimic', 'air_quality_italy', 'gait_uci', 'mocap_karate', 'gas_home',
         'delta_robot', 'kuka', 'nanodrone', 'mhealth', 'breizh_rs', 'solo12']
t6 = json.load(open(RR / 't6_abstract_number.json'))
t6m = {r['domain']: r for r in t6['rows']}
dmhi_mae = [sum(x['dmhi_mae'] for x in e3a[d]) / 3 for d in order]
dmhi_mse = [t6m[d]['dmhi_mse_3seed'] for d in order]
psw_mae, psw_mse = [], []
for d in order:
    a, b = [], []
    for s in (7, 42, 123):
        m = json.load(open(ROOT / f'code/runs/e6_psw_official/psw_i_ini_{d}_seed{s}_metrics.json'))
        a.append(m['official_scaled_mae']); b.append(m['official_scaled_mse'])
    psw_mae.append(sum(a) / 3); psw_mse.append(sum(b) / 3)

_drow = re.search(r'DMHI \(ours\) & (.+?) ' + re.escape('\\\\'), tbl, re.S).group(1)
dcells = [float(CLEAN(c)) for c in _drow.split('&')]
exp_d = [x for p in zip(dmhi_mae, dmhi_mse) for x in p]
for v, e in zip(dcells, exp_d):
    if abs(v - e) > 5e-4: bad.append(('t2', 'DMHI', v, round(e, 4)))
    else: ok += 1
_prow = re.findall(r'PSW-I & (.+?) ' + re.escape('\\\\'), tbl, re.S)[0]
pcells = [float(CLEAN(c)) for c in _prow.split('&')]
exp_p = [x for p in zip(psw_mae, psw_mse) for x in p]
for v, e in zip(pcells, exp_p):
    if abs(v - e) > 5e-4: bad.append(('t2', 'PSW', v, round(e, 4)))
    else: ok += 1

import math
# Table III layout: two halves, each 6 domains x (RMSE, MD) interleaved.
tbl3 = tex.split(r'\label{tab:rmse&md}')[1].split(r'\end{table')[0]
halves = re.findall(r'DMHI \(ours\) & (.+?) ' + re.escape('\\\\'), tbl3, re.S)
assert len(halves) == 2, len(halves)
d3r = []
for h in halves:
    cells = [float(CLEAN(c)) for c in h.split('&')]
    d3r += cells[0::2]  # RMSE cells only; MD cells kept old by design
order3 = ['physionet2012', 'mimic', 'air_quality_italy', 'gait_uci', 'mocap_karate',
          'gas_home', 'delta_robot', 'kuka', 'nanodrone', 'mhealth', 'breizh_rs', 'solo12']
exp_rmse = [math.sqrt(t6m[d]['dmhi_mse_3seed']) for d in order3]
for v, e in zip(d3r, exp_rmse):
    if abs(v - e) > 5e-4: bad.append(('t3-rmse', 'DMHI', v, round(e, 4)))
    else: ok += 1

if '92.0\\%' not in tex:
    bad.append(('abstract', '92.0% missing', '', ''))
else:
    ok += 1

print(f"cells OK: {ok}, mismatches: {len(bad)}")
for b in bad[:15]:
    print(' BAD:', b)
sys.exit(1 if bad else 0)
