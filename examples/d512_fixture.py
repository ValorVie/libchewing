# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generate tensor/score oracle for the synthetic cases with the external A1 harness."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--harness', type=Path, required=True)
ap.add_argument('--model-dir', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
spec = importlib.util.spec_from_file_location('d512eval', args.harness)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
scorer = h.Scorer(args.model_dir, threads=2)
raw_run = scorer.session.run
inputs = []


def record(outputs, feeds):
    inputs.append({k: dict(dtype=str(v.dtype), shape=list(v.shape), values=v.flatten().tolist())
                   for k, v in feeds.items()})
    return raw_run(outputs, feeds)


scorer.session.run = record
source = Path(__file__).resolve().parent.parent / 'tests/data/d512-requests.json'
cases = []
for r in json.loads(source.read_text())['cases']:
    inputs.clear()
    req = dict(r, candidates=[dict(text=t) for t in r['candidates']])
    assert h.fallback_reason(req, scorer) is None
    scores = scorer.score_request(req)
    cases.append(dict(request=r, inputs=list(inputs), scores=scores))
result = dict(harness_sha256=hashlib.sha256(args.harness.read_bytes()).hexdigest(),
              fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases)
with args.out.open('x') as f:
    json.dump(result, f, ensure_ascii=False, allow_nan=False)
print(f'{len(cases)} synthetic cases written')
