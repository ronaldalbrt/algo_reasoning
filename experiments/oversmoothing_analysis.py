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

"""Representation smoothness analysis for the SpectralMPNN revision (C3).

This is the post-hoc analysis script for Experiments 1 and 2 of the C3
remediation plan (``jmlr/c3_remediation_plan.md``). It quantifies how the
diversity of node embeddings evolves across algorithmic-execution steps, to
provide *direct* evidence for/against the oversmoothing hypothesis requested by
Reviewers 1 and 2.

Two metrics are computed on the per-step hidden node embeddings
``H^t in R^{N x d}`` returned by ``EncodeProcessDecode.forward`` (the
``hidden_embeddings`` tensor of shape ``[batch, steps, nodes, hidden]``):

* **Dirichlet energy** (Cai & Wang, 2020). Using the symmetric normalized
  Laplacian ``L = I - D^{-1/2} A D^{-1/2}`` of the encoded input graph, the
  (scale-invariant) normalized Dirichlet energy of a step is

      E(H) = trace(H^T L H) / trace(H^T H).

  Lower energy means neighbouring nodes share more similar embeddings, i.e.
  smoother / more collapsed representations. Progressive smoothing shows up as a
  decreasing curve over execution steps. The unnormalized ``trace(H^T L H)`` is
  also reported.

* **Mean Average Distance (MAD)**. The mean cosine distance between all distinct
  node pairs,

      MAD(H) = (2 / n(n-1)) * sum_{i<j} (1 - cos(h_i, h_j)).

  MAD collapsing toward 0 indicates representation collapse.

The expectation (the claim under test) is that MPNN embeddings lose diversity
across steps (decreasing E and MAD) while SpectralMPNN maintains higher
diversity. No retraining is required: the script loads existing checkpoints and
runs a single forward pass per model on a shared batch.

Example
-------
    python experiments/oversmoothing_analysis.py \
        --algorithm topological_sort --nb_nodes 64 --batch_size 32 \
        --model "mpnn:mpnn:checkpoints/topological_sort/topological_sort-mpnn0-step=09000-val_loss=0.50.ckpt" \
        --model "spectralmpnn:spectralmpnn:checkpoints/topological_sort/topological_sort-spectralmpnn0-step=09000-val_loss=0.40.ckpt" \
        --model "spectralmpnn_nomp:spectralmpnn:checkpoints/topological_sort/topological_sort-spectralmpnn0_NoMP-...ckpt:model_args/spectralmpnn_nomp.yaml"

Pass ``--model "label:processor:random"`` (checkpoint == ``random``) to smoke
test the pipeline with a randomly initialized model and no checkpoint on disk.
"""

import argparse
import json
import os

import torch
import yaml

from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.models.processor import normalized_laplacian
from algo_reasoning.src.sampler import SAMPLERS


def parse_model_spec(spec):
    """Parse a ``label:processor:checkpoint[:model_args_yaml]`` CLI entry.

    The checkpoint path itself may contain ``:`` characters (lightning encodes
    ``step=09000`` etc.), so only the first two ``:`` separate the label and the
    processor; the remainder is the checkpoint, optionally followed by a single
    trailing ``::model_args.yaml`` style suffix is *not* supported -- use the
    ``model_args`` separator explicitly via the 4th field below.
    """
    parts = spec.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            f"--model expects 'label:processor:checkpoint[:model_args_yaml]', got {spec!r}"
        )
    label, processor = parts[0], parts[1]
    # A model_args yaml (if any) is the last field and ends with .yaml/.yml.
    model_args = None
    rest = parts[2:]
    if rest[-1].endswith((".yaml", ".yml")) and len(rest) > 1:
        model_args = rest[-1]
        checkpoint = ":".join(rest[:-1])
    else:
        checkpoint = ":".join(rest)
    return {"label": label, "processor": processor,
            "checkpoint": checkpoint, "model_args": model_args}


def load_model(spec, algorithm, nb_triplet_fts, device):
    """Build an ``EncodeProcessDecode`` and load weights from a checkpoint."""
    model_args = {}
    if spec["model_args"]:
        with open(spec["model_args"], "r") as f:
            model_args = yaml.safe_load(f) or {}

    model = EncodeProcessDecode(
        [algorithm],
        processor=spec["processor"],
        nb_triplet_fts=nb_triplet_fts if nb_triplet_fts > 0 else None,
        **model_args,
    )

    if spec["checkpoint"] != "random":
        ckpt = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Lightning stores parameters under a 'model.' prefix.
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[len("model."):] if k.startswith("model.") else k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[{spec['label']}] WARNING: {len(missing)} missing keys "
                  f"(e.g. {missing[:3]})")
        if unexpected:
            print(f"[{spec['label']}] WARNING: {len(unexpected)} unexpected keys "
                  f"(e.g. {unexpected[:3]})")
    else:
        print(f"[{spec['label']}] using RANDOM initialization (no checkpoint).")

    return model.to(device).eval()


def dirichlet_energy(H, L, eps=1e-9):
    """Normalized and raw Dirichlet energy.

    Args:
        H: hidden embeddings, shape ``[B, N, d]``.
        L: (normalized) Laplacian, shape ``[B, N, N]``.

    Returns:
        ``(normalized, raw)`` tensors of shape ``[B]`` (per-sample energies).
    """
    raw = torch.einsum("bnf,bnm,bmf->b", H, L, H)
    norm = torch.einsum("bnf,bnf->b", H, H)
    return raw / (norm + eps), raw


def mean_average_distance(H, eps=1e-9):
    """MAD over all distinct node pairs. Returns per-sample tensor ``[B]``."""
    Hn = torch.nn.functional.normalize(H, dim=-1, eps=eps)
    cos = torch.matmul(Hn, Hn.transpose(-1, -2))          # [B, N, N]
    dist = 1.0 - cos
    n = H.shape[1]
    off_diag = dist.sum(dim=(-1, -2)) - torch.diagonal(dist, dim1=-1, dim2=-2).sum(-1)
    denom = max(n * (n - 1), 1)
    return off_diag / denom


@torch.no_grad()
def analyze(model, batch, algorithm, device):
    """Run a forward pass and compute per-step Dirichlet energy and MAD."""
    batch = batch.to(device)

    output = model(batch)
    hidden = output["hidden_embeddings"]                  # [B, steps, N, d]

    # Adjacency of the encoded input graph (symmetric, with self-loops as built
    # by the encoder). No hints are fed, so this is the step-0 input structure,
    # held fixed across steps to define a single reference Laplacian.
    _, _, _, adj = model.encoders[algorithm](batch)
    L = normalized_laplacian(adj)                         # [B, N, N]

    n_steps = hidden.shape[1]
    per_step = []
    for t in range(n_steps):
        H_t = hidden[:, t]                                # [B, N, d]
        e_norm, e_raw = dirichlet_energy(H_t, L)
        mad = mean_average_distance(H_t)
        per_step.append({
            "step": t,
            "dirichlet_energy_normalized": e_norm.mean().item(),
            "dirichlet_energy_raw": e_raw.mean().item(),
            "mad": mad.mean().item(),
        })
    return per_step


def maybe_plot(results, algorithm, output_dir):
    """Plot energy and MAD curves if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, per_step in results.items():
        steps = [r["step"] for r in per_step]
        axes[0].plot(steps, [r["dirichlet_energy_normalized"] for r in per_step],
                     marker="o", label=label)
        axes[1].plot(steps, [r["mad"] for r in per_step], marker="o", label=label)

    axes[0].set_title(f"Normalized Dirichlet energy -- {algorithm}")
    axes[0].set_xlabel("Execution step")
    axes[0].set_ylabel(r"$E(H) = \mathrm{tr}(H^\top L H)/\mathrm{tr}(H^\top H)$")
    axes[1].set_title(f"Mean Average Distance -- {algorithm}")
    axes[1].set_xlabel("Execution step")
    axes[1].set_ylabel("MAD")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"oversmoothing_{algorithm}.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", required=True,
                    help=f"CLRS algorithm. One of: {sorted(SAMPLERS)}")
    ap.add_argument("--model", action="append", dest="models", required=True,
                    type=parse_model_spec,
                    help="Repeatable. 'label:processor:checkpoint[:model_args_yaml]'. "
                         "Use checkpoint='random' for an untrained smoke test.")
    ap.add_argument("--nb_nodes", type=int, default=64,
                    help="Graph size to sample (test-time size; default 64).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--nb_triplet_fts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None,
                    help="cpu/cuda/mps. Default: auto-detect.")
    ap.add_argument("--algorithms_args", default="algorithm_args/default.yaml",
                    help="YAML with per-algorithm sampling kwargs (or '' for none).")
    ap.add_argument("--output_dir", default="experiments/oversmoothing_results")
    args = ap.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.algorithm not in SAMPLERS:
        raise SystemExit(f"Unknown algorithm {args.algorithm!r}; "
                         f"choose from {sorted(SAMPLERS)}")

    # Per-algorithm sampling kwargs (e.g. edge probabilities), matching run.py.
    algo_kwargs = {}
    if args.algorithms_args:
        with open(args.algorithms_args, "r") as f:
            loaded = yaml.safe_load(f) or {}
        algo_kwargs = loaded.get(args.algorithm, {})

    # A single shared batch so every model sees identical graphs.
    nb_nodes = 4 if args.algorithm in ("segments_intersect", "carls_vacation") else args.nb_nodes
    generator = torch.Generator().manual_seed(args.seed)
    sampler = SAMPLERS[args.algorithm](generator=generator, randomize_pos=False)
    batch = sampler.sample(nb_nodes, args.batch_size, **algo_kwargs)

    results = {}
    for spec in args.models:
        print(f"\n=== {spec['label']} ({spec['processor']}) ===")
        model = load_model(spec, args.algorithm, args.nb_triplet_fts, device)
        per_step = analyze(model, batch, args.algorithm, device)
        results[spec["label"]] = per_step
        for r in per_step:
            print(f"  step {r['step']:2d}  E_norm={r['dirichlet_energy_normalized']:.4f}  "
                  f"E_raw={r['dirichlet_energy_raw']:.2f}  MAD={r['mad']:.4f}")

    payload = {
        "algorithm": args.algorithm,
        "nb_nodes": nb_nodes,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "results": results,
    }
    json_path = os.path.join(args.output_dir, f"oversmoothing_{args.algorithm}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    maybe_plot(results, args.algorithm, args.output_dir)


if __name__ == "__main__":
    main()
