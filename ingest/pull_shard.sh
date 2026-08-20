#!/bin/bash
# Pull a finished shard from the GPU box over LAN HTTP.
# Usage: bash pull_shard.sh as
set -euo pipefail
lang="${1:?language code}"
base="http://172.17.15.250:8766/${lang}"
dest="/vm-storage/hhrag/indexes/${lang}"
mkdir -p "$dest"
if ! curl -fsSI --max-time 10 "${base}/shard_meta.json" >/dev/null; then
  echo "no shard_meta yet for ${lang}"
  exit 2
fi
for f in shard_meta.json dense_meta.json chunks.offsets.npy faiss.index chunks.jsonl; do
  echo "get ${f}"
  curl -fL --retry 5 -C - -o "${dest}/${f}" "${base}/${f}"
done
python3 - <<PY
import json
from pathlib import Path
p = Path("${dest}") / "shard_meta.json"
print(p.read_text()[:400])
PY
echo "pulled ${lang} -> ${dest}"
