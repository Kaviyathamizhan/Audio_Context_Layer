"""Edge-feasibility numbers for the detector: parameters, size, CPU latency, dynamic INT8 quantisation.

  python -m acl.edge_report --data dataset --run runs/crnn

For the AST backend only the trainable head is small; the frozen AST backbone (~87 M params) would have to be
distilled/compressed for on-device use - reported here so the trade-off is explicit.
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import load_annotations, load_waves
from .metrics import event_f1
from .models import build_model, n_params
from .timeline import decode


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def latency_ms(model, x, n=20):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        ts = []
        for _ in range(n):
            t = time.perf_counter(); model(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--run", default="runs/crnn")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.load(open(run / "config.json"))
    torch.set_num_threads(1)
    res = {"backend": cfg["backend"], "trainable_params": cfg["params"]}
    if cfg["backend"] == "ast":
        from transformers import ASTConfig, ASTModel
        res["frozen_ast_backbone_params"] = n_params(ASTModel(ASTConfig()))
        res["note"] = "AST backbone is frozen and too large for a watch-class device; distil or replace it for deployment."
        json.dump(res, open(run / "edge.json", "w"), indent=1)
        print(json.dumps(res, indent=1)); return

    model = build_model("crnn")
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    x1 = torch.randn(1, 160000) * 0.05
    res.update({"fp32_size_mb": size_mb(model), "cpu_latency_ms_per_10s_fp32_1thread": latency_ms(model, x1)})
    res["real_time_factor_fp32"] = res["cpu_latency_ms_per_10s_fp32_1thread"] / 10000.0
    try:
        from torch.ao.quantization import quantize_dynamic
        q = quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        res.update({"int8_dynamic_size_mb": size_mb(q), "cpu_latency_ms_per_10s_int8_1thread": latency_ms(q, x1)})
        # accuracy impact: event-F1 fp32 vs int8 on test using the tuned decode params
        ann = load_annotations(a.data, "test", a.limit)
        w = torch.from_numpy(load_waves(a.data, ann)).float() / 32768.0
        rj = json.load(open(run / "results" / "results.json"))
        dp = rj["decode_params"]
        gt = [x["events"] for x in ann]

        def f1_of(m):
            with torch.no_grad():
                p = torch.cat([torch.sigmoid(m(w[i:i + 32])) for i in range(0, len(w), 32)]).numpy()
            return event_f1([decode(pp, **dp) for pp in p], gt)[0]

        res["event_f1_fp32"], res["event_f1_int8_dynamic"] = f1_of(model), f1_of(q)
    except Exception as e:  # quantisation backend may be unavailable on some platforms
        res["int8_error"] = repr(e)
    json.dump(res, open(run / "edge.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
