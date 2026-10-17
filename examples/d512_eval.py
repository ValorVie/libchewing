"""Evaluate explicit libchewing JSONL with the unmodified external A1 harness.

Run dev first for each cap, then held-out with its frozen selection JSON.
No corpus, model, harness implementation, or machine paths are embedded here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
from pathlib import Path

# Includes A1's original grid and the finer grid used for its final selection.
WEIGHTS = (0, .25, .5, .75, 1, 1.25, 1.5, 2, 3, 5, 8)
MARGINS = (0, .25, .5, 1, 1.5, 2)


def select_point(grid):
    valid = [p for p in grid if 3*p['harmful_overrides'] <= p['helpful_overrides']]
    if not valid:
        raise ValueError('no point meets the harmful/helpful constraint')
    return min(valid, key=lambda p: (-p['delta_pp'], p['w'], p['tau']))


def select_presets(grid):
    """Select from dev aggregates only; keep the original balanced point.

    Conservative halves balanced's harmful/helpful ratio and retains positive
    gain. Both adjustable presets maximize gain, with the existing tie order.
    """
    balanced = next((p for p in grid if (p['w'], p['tau']) == (.25, .25)), None)
    if balanced is None or balanced['helpful_overrides'] == 0:
        raise ValueError('missing usable frozen balanced point')
    conservative = [p for p in grid
                    if p['helpful_overrides'] > p['harmful_overrides']
                    and 2 * p['harmful_overrides'] * balanced['helpful_overrides']
                    <= balanced['harmful_overrides'] * p['helpful_overrides']]
    return dict(conservative=select_point(conservative), balanced=balanced,
                aggressive=select_point(grid))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    # Exclusive create prevents silently overwriting a held-out result.
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=1, allow_nan=False)


def latency(values):
    if not values:
        return None
    q = sorted(values)
    return dict(n=len(q), median_ms=statistics.median(q),
                p95_ms=q[min(len(q)-1, int(.95*len(q)))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['dev', 'held-out'])
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--harness', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--cap', type=int, choices=[8, 10], required=True)
    args = ap.parse_args()
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'
    spec = importlib.util.spec_from_file_location('d512eval', args.harness)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    import onnxruntime as ort
    if ort.__version__ != '1.31.0':
        raise ValueError('requires A1 ONNX Runtime 1.31.0')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / f'{args.stage}-{args.cap}'
    result_path = prefix.with_suffix('.result.json')
    if result_path.exists():
        raise FileExistsError(result_path)
    selection_path = args.out_dir / f'selection-{args.cap}.json'
    frozen = None
    if args.stage == 'held-out':
        frozen = json.loads(selection_path.read_text())
        if frozen['harness_sha256'] != digest(args.harness) or frozen['cap'] != args.cap:
            raise ValueError('frozen selection/harness mismatch')
    reqs = h.load_requests(args.requests)
    for r in reqs:
        r['candidates'] = r['candidates'][:args.cap]
        r['gold_in_candidates'] = any(c['text'] == r['gold'] for c in r['candidates'])
    if not reqs or len({r['id'] for r in reqs}) != len(reqs):
        raise ValueError('empty input or duplicate request id')
    scorer = h.Scorer(args.model_dir, threads=2)
    pins = {p: digest(args.model_dir/p) for p in ['model.onnx', 'vocab.json', 'manifest.json']}
    if frozen and frozen['model_sha256'] != pins:
        raise ValueError('frozen model mismatch')
    scored = h.score_all(reqs, scorer, warmup=3)
    # Persist raw scores before evaluation so expensive inference is auditable.
    write_json(prefix.with_suffix('.scores.json'), [dict(id=r['id'], **s) for r, s in zip(reqs, scored)])
    meta = dict(cap=args.cap, requests_sha256=digest(args.requests),
                harness_sha256=digest(args.harness), model_sha256=pins,
                ort=ort.__version__, providers=scorer.session.get_providers(),
                intra_op=2, warmup=3, machine=platform.platform(),
                python=platform.python_version(), weights=WEIGHTS, margins=MARGINS)
    grid = None
    if args.stage == 'dev':
        grid = [h.evaluate(reqs, scored, w, t, bootstrap=400) for w in WEIGHTS for t in MARGINS]
        point = select_point(grid)
        write_json(selection_path, dict(**meta, point=point, grid=grid,
                                      dev_ids=[r['id'] for r in reqs]))
    else:
        if {r['id'] for r in reqs} & set(frozen['dev_ids']):
            raise ValueError('dev/held-out overlap')
        point = frozen['point']
    w, tau = point['w'], point['tau']
    result = h.evaluate(reqs, scored, w, tau, bootstrap=1000)
    subsets = {}
    groups = {
        'length_2_4': lambda r: 2 <= len(r['reading']) <= 4,
        'length_5_8': lambda r: 5 <= len(r['reading']) <= 8,
        'length_9_16': lambda r: 9 <= len(r['reading']) <= 16,
        'length_17_32': lambda r: 17 <= len(r['reading']) <= 32,
        'wikipedia': lambda r: '維基百科' in r.get('source', ''),
        'chat': lambda r: '維基百科' not in r.get('source', ''),
    }
    for name, includes in groups.items():
        indices = [i for i, r in enumerate(reqs) if includes(r)]
        if indices:
            subsets[name] = h.evaluate([reqs[i] for i in indices], [scored[i] for i in indices], w, tau, bootstrap=1000)
    harmful = []
    for r, s in zip(reqs, scored):
        d = r['engine_default']
        k = h.choose([c['log_score'] for c in r['candidates']], s['model'], d, w, tau)
        if r['candidates'][d]['text'] == r['gold'] and r['candidates'][k]['text'] != r['gold']:
            harmful.append(dict(id=r['id'], gold=r['gold'], baseline=r['candidates'][d]['text'],
                                selected=r['candidates'][k]['text']))
    summary = h.summarize(reqs, scored, scorer)
    summary['engine_latency'] = latency([r['engine_ms'] for r in reqs])
    summary['candidate_counts'] = {str(n): sum(len(r['candidates']) == n for r in reqs) for n in range(1, args.cap+1)}
    write_json(result_path, dict(meta=meta, selected=result, summary=summary, subsets=subsets,
                                harmful=harmful, grid=grid))
    print(json.dumps(dict(stage=args.stage, cap=args.cap, selected=result, summary=summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
