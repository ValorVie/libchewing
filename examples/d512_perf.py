# SPDX-License-Identifier: LGPL-2.1-or-later
"""Replay the same synthetic tensor oracle as d512_perf.rs (run sequentially).

Usage: python d512_perf.py MODEL_DIR ORACLE_JSON OUTPUT_JSON [PROFILE_PREFIX]
Requires the existing numpy/onnxruntime environment; downloads nothing.
Set ORT_DISABLE_TELEMETRY=1 before launch. Profiling affects timings.
"""
import json
import os
from pathlib import Path
import sys
import time

assert os.environ.get('ORT_DISABLE_TELEMETRY') == '1'
import numpy as np
import onnxruntime as ort

assert 4 <= len(sys.argv) <= 5, __doc__
model, oracle, out = map(Path, sys.argv[1:4])
fixture = json.loads(oracle.read_text())
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 1
opts.log_severity_level = 1
if len(sys.argv) == 5:
    opts.enable_profiling = True
    opts.profile_file_prefix = sys.argv[4]
session = ort.InferenceSession(str(model / 'model.onnx'), opts, providers=['CPUExecutionProvider'])
rows = []
for case_index, case in enumerate(fixture['cases']):
    for iteration in range(13):
        phases = [0.0] * 3
        scores = []
        for x in case['inputs']:
            start = time.perf_counter()
            feeds = {name: np.asarray(t['values'], dtype=t['dtype']).reshape(t['shape'])
                     for name, t in x.items() if name != 'mask'}
            length = x['mask']['shape'][2]
            feeds['mask'] = np.tril(np.ones((length, length), dtype=bool))[None, None]
            assert feeds['mask'].flatten().tolist() == x['mask']['values']
            phases[0] += (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            result = session.run(['scores'], feeds)
            phases[1] += (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            scores.append(float(result[0][0]))
            del result
            phases[2] += (time.perf_counter() - start) * 1000
        if iteration >= 3:
            rows.append(dict(case=case_index, iteration=iteration-3, tensor_ms=phases[0],
                             run_ms=phases[1], extract_ms=phases[2], scores=scores))
profile = session.end_profiling() if len(sys.argv) == 5 else None
properties = ('intra_op_num_threads', 'inter_op_num_threads', 'execution_mode',
              'graph_optimization_level', 'enable_mem_pattern', 'enable_cpu_mem_arena',
              'use_deterministic_compute')
out.write_text(json.dumps(dict(build=ort.get_build_info(), profile=profile,
                               options={k: str(getattr(opts, k)) for k in properties}, rows=rows), indent=2))
