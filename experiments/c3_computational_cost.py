"""Computational-cost benchmark for the SpectralMPNN JMLR revision (Section 5.6).

Addresses reviewer concern **M3** (R2-M3, R3-M4, R3-Q2): report wall-clock
training/inference time and memory for MPNN vs SpectralMPNN vs PolySpectralMPNN,
and expose the O(n^3) eigendecomposition bottleneck of SpectralMPNN by scaling
graph size beyond the 64-node test regime.

RUN THIS ON THE RTX 3090 (not the dev box). It does NOT need trained
checkpoints -- latency/throughput/memory depend on architecture and graph size,
not on learned weights, so models are randomly initialized. (Pass --checkpoints
if you want to time the exact trained models instead; timing is unchanged.)

What it measures, per processor x graph size n:
  * Inference latency  : median ms per forward pass (no_grad), + peak GPU memory.
  * Training step time : median ms per forward+backward+optimizer step, + peak mem.
  * Parameter count    : total trainable params (answers R3-m4 on FFN capacity).
Failures are recorded, not fatal: CUDA OOM -> "OOM"; eigh non-convergence on a
degenerate graph -> "eigh_fail" (SpectralMPNN only). This is expected at large n
and is itself part of the scalability story.

Outputs (into --output_dir, default experiments/cost_results):
  * cost_benchmark.json  : all raw numbers.
  * cost_latency.png     : inference + training latency vs n (log-log).
  * cost_memory.png      : peak GPU memory vs n.
  * a Markdown table printed to stdout (paste-ready for Section 5.6).

Example:
  python experiments/c3_computational_cost.py \
      --algorithm bfs --batch_size 32 \
      --nb_nodes_sweep 16,32,64,128,256 --repeats 30 --warmup 10 --device cuda

Notes:
  * n=256 (or 128) may OOM SpectralMPNN on a 24GB 3090 -- that is the point of
    the experiment; the table/plot will show it as OOM while MPNN/Poly continue.
  * Pick a graph-structured algorithm (bfs/dfs/dijkstra/bellman_ford). Sorting/DP
    tasks have degenerate graphs and are less representative of the eigendecomp cost.
"""

import argparse
import json
import os
import statistics
import time

import torch
from torch.optim import AdamW

from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.losses.AlgorithmicReasoningLoss import AlgorithmicReasoningLoss
from algo_reasoning.src.sampler import SAMPLERS

try:
    import yaml
except ImportError:
    yaml = None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mib(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def _is_eigh_failure(exc):
    m = str(exc)
    return ("eigh" in m) or ("LinAlg" in m) or ("converge" in m)


def build_batch(algorithm, n, batch_size, seed, algo_kwargs):
    # Some algorithms force a fixed node count; respect it.
    nb_nodes = 4 if algorithm in ("segments_intersect", "carls_vacation") else n
    gen = torch.Generator().manual_seed(seed)
    sampler = SAMPLERS[algorithm](generator=gen, randomize_pos=False)
    return sampler.sample(nb_nodes, batch_size, **algo_kwargs), nb_nodes


def time_inference(model, batch, device, repeats, warmup):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        _sync(device)
        _reset_peak(device)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(batch)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), _peak_mib(device)


def time_training(model, batch, loss_fn, device, repeats, warmup):
    model.train()
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def one_step():
        optim.zero_grad(set_to_none=True)
        out = model(batch.clone())
        loss = loss_fn(out["output"], batch, out["hidden_embeddings"])
        loss.backward()
        optim.step()
        return loss

    for _ in range(warmup):
        one_step()
    _sync(device)
    _reset_peak(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), _peak_mib(device)


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", default="bfs",
                    help="CLRS algorithm (graph-structured recommended).")
    ap.add_argument("--processors", default="mpnn,spectralmpnn,polyspectralmpnn",
                    help="Comma list of processors to compare.")
    ap.add_argument("--nb_nodes_sweep", default="16,32,64,128,256",
                    help="Comma list of graph sizes n.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--nb_triplet_fts", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None, help="cuda/cpu. Default: auto.")
    ap.add_argument("--algorithms_args", default="algorithm_args/default.yaml",
                    help="YAML with per-algorithm sampling kwargs (or '' for none).")
    ap.add_argument("--output_dir", default="experiments/cost_results")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.algorithm not in SAMPLERS:
        raise SystemExit(f"Unknown algorithm {args.algorithm!r}; choose from {sorted(SAMPLERS)}")

    algo_kwargs = {}
    if args.algorithms_args and yaml is not None:
        with open(args.algorithms_args, "r") as f:
            algo_kwargs = (yaml.safe_load(f) or {}).get(args.algorithm, {})

    processors = [p.strip() for p in args.processors.split(",") if p.strip()]
    sizes = [int(x) for x in args.nb_nodes_sweep.split(",")]
    nb_triplet = args.nb_triplet_fts if args.nb_triplet_fts > 0 else None
    loss_fn = AlgorithmicReasoningLoss(lr=1e-3, weight_decay=1e-2, reg_weight=0.0)

    if device.type == "cuda":
        print(f"# GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"# algorithm={args.algorithm}  batch_size={args.batch_size}  "
          f"repeats={args.repeats}  warmup={args.warmup}\n")

    results = {"meta": {"algorithm": args.algorithm, "batch_size": args.batch_size,
                        "repeats": args.repeats, "warmup": args.warmup,
                        "device": (torch.cuda.get_device_name(0)
                                   if device.type == "cuda" else "cpu"),
                        "sizes": sizes, "processors": processors},
               "runs": []}

    for proc in processors:
        for n in sizes:
            rec = {"processor": proc, "n": n}
            try:
                batch, nb_nodes = build_batch(args.algorithm, n, args.batch_size,
                                              args.seed, algo_kwargs)
                batch = batch.to(device)
                rec["nb_nodes"] = nb_nodes
                model = EncodeProcessDecode([args.algorithm], processor=proc,
                                            seed=args.seed, nb_triplet_fts=nb_triplet).to(device)
                rec["params"] = param_count(model)

                try:
                    inf_ms, inf_mem = time_inference(model, batch, device,
                                                     args.repeats, args.warmup)
                    rec["infer_ms"] = round(inf_ms, 3)
                    rec["infer_peak_mib"] = round(inf_mem, 1)
                except torch.cuda.OutOfMemoryError:
                    rec["infer_ms"] = "OOM"; rec["infer_peak_mib"] = "OOM"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    rec["infer_ms"] = "eigh_fail" if _is_eigh_failure(e) else f"ERR:{type(e).__name__}"
                    rec["infer_peak_mib"] = rec["infer_ms"]

                try:
                    tr_ms, tr_mem = time_training(model, batch, loss_fn, device,
                                                  args.repeats, args.warmup)
                    rec["train_ms"] = round(tr_ms, 3)
                    rec["train_peak_mib"] = round(tr_mem, 1)
                except torch.cuda.OutOfMemoryError:
                    rec["train_ms"] = "OOM"; rec["train_peak_mib"] = "OOM"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    rec["train_ms"] = "eigh_fail" if _is_eigh_failure(e) else f"ERR:{type(e).__name__}"
                    rec["train_peak_mib"] = rec["train_ms"]

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                rec["infer_ms"] = rec["train_ms"] = "OOM"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"

            results["runs"].append(rec)
            print(f"{proc:18s} n={n:<4} params={rec.get('params','?'):>8} "
                  f"infer={rec.get('infer_ms','?')} ms ({rec.get('infer_peak_mib','?')} MiB)  "
                  f"train={rec.get('train_ms','?')} ms ({rec.get('train_peak_mib','?')} MiB)")

    out_json = os.path.join(args.output_dir, "cost_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    print_markdown(results)
    maybe_plot(results, args.output_dir)


def print_markdown(results):
    procs = results["meta"]["processors"]
    sizes = results["meta"]["sizes"]
    idx = {(r["processor"], r["n"]): r for r in results["runs"]}
    print("\n### Inference latency (ms/batch) | training step (ms) | peak train mem (MiB)\n")
    header = "| n | " + " | ".join(procs) + " |"
    print(header)
    print("|" + "---|" * (len(procs) + 1))
    for n in sizes:
        cells = []
        for p in procs:
            r = idx.get((p, n), {})
            cells.append(f"{r.get('infer_ms','-')} / {r.get('train_ms','-')} / {r.get('train_peak_mib','-')}")
        print(f"| {n} | " + " | ".join(str(c) for c in cells) + " |")
    print("\nParameter counts:",
          {p: next((r.get("params") for r in results["runs"] if r["processor"] == p), None)
           for p in procs})


def maybe_plot(results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return
    procs = results["meta"]["processors"]
    idx = {(r["processor"], r["n"]): r for r in results["runs"]}

    def series(proc, key):
        xs, ys = [], []
        for n in results["meta"]["sizes"]:
            v = idx.get((proc, n), {}).get(key)
            if isinstance(v, (int, float)):
                xs.append(n); ys.append(v)
        return xs, ys

    # latency
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for proc in procs:
        for ax, key, title in ((axes[0], "infer_ms", "Inference latency"),
                               (axes[1], "train_ms", "Training step time")):
            xs, ys = series(proc, key)
            if xs:
                ax.plot(xs, ys, marker="o", label=proc)
    for ax, title in ((axes[0], "Inference latency"), (axes[1], "Training step time")):
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("graph size n"); ax.set_ylabel("ms/batch")
        ax.set_title(title); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(output_dir, "cost_latency.png")
    fig.savefig(p1, dpi=200); print(f"Saved {p1}")

    # memory
    fig2, ax = plt.subplots(figsize=(6, 4))
    for proc in procs:
        xs, ys = series(proc, "train_peak_mib")
        if xs:
            ax.plot(xs, ys, marker="o", label=proc)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("graph size n"); ax.set_ylabel("peak GPU memory (MiB)")
    ax.set_title("Peak training memory"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(output_dir, "cost_memory.png")
    fig2.savefig(p2, dpi=200); print(f"Saved {p2}")


if __name__ == "__main__":
    main()
