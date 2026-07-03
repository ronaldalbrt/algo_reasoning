# Copyright (C) 2024 Ronald Albert ronaldalbert@cos.ufrj.br
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wall-clock timing / scalability analysis for the SpectralMPNN revision (M3).

Addresses reviewer concerns M3 / R2-M3 / R3-M4 / R3-Q2: report wall-clock
training and inference times for MPNN, SpectralMPNN, and PolynomialSpectralMPNN,
and characterize the O(n^3) eigendecomposition bottleneck of SpectralMPNN by
sweeping the graph size n.

No checkpoints needed: timing depends on architecture + shapes, not learned
weights, so freshly initialized models are sufficient. (Pass checkpoints only if
you want timings for the exact trained configs.)

Two measurements per (model, n):
  * inference time  — `model(batch)` under no_grad, per batch.
  * training step   — `model(batch)` + loss + `loss.backward()`, per batch
    (a proxy for cost/step; multiply by steps/epoch for per-epoch time).

Reports mean +/- std over timed iterations (after warmup), in milliseconds per
batch, plus peak GPU memory (MiB) if CUDA is used. CUDA timings are wrapped in
`torch.cuda.synchronize()`.

Example
-------
    python experiments/timing_analysis.py \
        --algorithm dijkstra --batch_size 32 \
        --nb_nodes 16,32,64,128 \
        --models mpnn,spectralmpnn,polyspectralmpnn \
        --warmup 3 --iters 20 --device cuda
"""

import argparse
import json
import os
import statistics
import time

import torch

from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.sampler import SAMPLERS
from algo_reasoning.src.losses.AlgorithmicReasoningLoss import AlgorithmicReasoningLoss


def list_of_strings(arg):
    return [s.strip() for s in arg.split(",") if s.strip()]


def list_of_ints(arg):
    return [int(s) for s in arg.split(",") if s.strip()]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_calls(fn, warmup, iters, device):
    """Run fn() warmup times then time it iters times. Returns list of ms."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(times):
    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "n": len(times),
    }


def build(processor, algorithm, nb_triplet_fts, device):
    model = EncodeProcessDecode(
        [algorithm], processor=processor,
        nb_triplet_fts=nb_triplet_fts if nb_triplet_fts > 0 else None,
    )
    return model.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", default="dijkstra",
                    help=f"CLRS algorithm to sample timing batches from. One of {sorted(SAMPLERS)}")
    ap.add_argument("--models", type=list_of_strings,
                    default=["mpnn", "spectralmpnn", "polyspectralmpnn"])
    ap.add_argument("--nb_nodes", type=list_of_ints, default=[16, 32, 64, 128],
                    help="Comma-separated graph sizes to sweep (shows O(n^3) for SpectralMPNN).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--nb_triplet_fts", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None, help="cpu/cuda/mps. Default: auto.")
    ap.add_argument("--output_dir", default="experiments/timing_results")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.algorithm not in SAMPLERS:
        raise SystemExit(f"Unknown algorithm {args.algorithm!r}")

    loss_fn = AlgorithmicReasoningLoss()
    results = {}

    for nb_nodes in args.nb_nodes:
        n = 4 if args.algorithm in ("segments_intersect", "carls_vacation") else nb_nodes
        gen = torch.Generator().manual_seed(args.seed)
        sampler = SAMPLERS[args.algorithm](generator=gen, randomize_pos=False)
        batch = sampler.sample(n, args.batch_size).to(device)
        results.setdefault(n, {})

        for proc in args.models:
            model = build(proc, args.algorithm, args.nb_triplet_fts, device)

            # Inference timing.
            model.eval()
            def infer():
                with torch.no_grad():
                    model(batch.clone())
            try:
                inf = summarize(time_calls(infer, args.warmup, args.iters, device))
            except Exception as e:  # e.g. eigh non-convergence on a degenerate batch
                inf = {"error": str(e)}

            # Training-step timing (forward + loss + backward).
            model.train()
            def train_step():
                model.zero_grad(set_to_none=True)
                out = model(batch.clone())
                loss = loss_fn(out["output"], batch, out["hidden_embeddings"])
                loss.backward()
            try:
                trn = summarize(time_calls(train_step, args.warmup, args.iters, device))
            except Exception as e:
                trn = {"error": str(e)}

            mem = None
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                torch.cuda.reset_peak_memory_stats()

            # Trajectory length (number of execution steps), so timings can be
            # normalized per step -> algorithm-independent per-step cost.
            n_steps = None
            try:
                with torch.no_grad():
                    n_steps = int(model(batch.clone())["hidden_embeddings"].shape[1])
            except Exception:
                pass

            def add_per_step(summary):
                if isinstance(summary, dict) and "mean_ms" in summary and n_steps:
                    summary["per_step_ms"] = summary["mean_ms"] / n_steps
                    summary["n_steps"] = n_steps
                return summary

            add_per_step(inf)
            add_per_step(trn)

            results[n][proc] = {"inference": inf, "train_step": trn,
                                "peak_mem_mib": mem, "n_steps": n_steps}
            tag = inf.get("mean_ms")
            tag = f"{tag:.1f}ms" if isinstance(tag, float) else "ERR"
            ps = inf.get("per_step_ms")
            ps = f" ({ps:.2f}ms/step)" if isinstance(ps, float) else ""
            print(f"n={n:4d} {proc:18s} infer={tag}{ps} "
                  f"train={trn.get('mean_ms', float('nan')):.1f}ms "
                  f"steps={n_steps} mem={mem if mem is None else round(mem)}MiB")

    payload = {
        "algorithm": args.algorithm,
        "batch_size": args.batch_size,
        "device": str(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    out = os.path.join(args.output_dir, f"timing_{args.algorithm}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved timing to {out}")


if __name__ == "__main__":
    main()
