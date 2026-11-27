#!/bin/bash
# Build the Windows-safe publish package for 王笳辉 review.
# Output: /Users/thiswind/Desktop/_临时下载/DMHI-R1-回复素材-胡矿{,.zip}
set -euo pipefail
SRC_MD="paper/revision/回复素材-胡矿.md"
OUT_DIR="/Users/thiswind/Desktop/_临时下载/DMHI-R1-回复素材-胡矿"
OUT_ZIP="$OUT_DIR.zip"

rm -rf "$OUT_DIR" "$OUT_ZIP"
mkdir -p "$OUT_DIR/assets"
python3 - "$SRC_MD" "$OUT_DIR/回复素材-胡矿.md" <<'PYEOF'
import sys, re
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t = src.read_text()
# rewrite bare image refs into assets/
t = re.sub(r'\]\((e4_[a-zA-Z_0-9]+-1\.png)\)', r'](assets/\1)', t)
dst.write_text(t)
imgs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', t)
assert all((dst.parent / i).name for i in imgs)
print('image refs rewritten:', len(imgs))
PYEOF
cp paper/revision/e4_r_L_preview-1.png paper/revision/e4_k_preview-1.png \
   paper/revision/e4_gamma_s_preview-1.png paper/revision/e4_d_preview-1.png "$OUT_DIR/assets/"
cp code/results/revision-r1/e4_sensitivity/e4_sensitivity_*.pdf "$OUT_DIR/assets/"
# remove Finder junk if any
find "$OUT_DIR" -name .DS_Store -delete

python3 - "$OUT_ZIP" "$OUT_DIR" <<'PYEOF'
import sys, zipfile
from pathlib import Path
out, src = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(src.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(src.parent))
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    cjk = [i for i in z.infolist() if any(ord(c) > 127 for c in i.filename)]
    assert all(i.flag_bits & 0x800 for i in cjk), 'UTF-8 flag missing'
    print('zip ok:', len(z.namelist()), 'entries, all CJK names UTF-8-flagged')
PYEOF
echo "built: $OUT_ZIP ($(stat -f%z "$OUT_ZIP") bytes)"
