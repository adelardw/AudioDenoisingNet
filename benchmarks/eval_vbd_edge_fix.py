"""VoiceBank+DEMAND test set: noisy input vs. original inference vs. edge-fixed inference.

    python benchmarks/eval_vbd_edge_fix.py --ckpt ckpts/ultra/checkpoints/last.ckpt --out vbd_edge_fix.csv

Prints mean SI-SDR / PESQ-WB / PESQ-NB / STOI per variant and paired differences with 95% CIs.
By default the 824-clip test split is downloaded from the Hugging Face Hub; ``--data`` also accepts
a local parquet file with ``noisy`` / ``clean`` audio columns.
"""
import argparse
import csv
import io
import os
import sys
import types

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # mamba_ssm is CUDA-only and only needed by the Mamba variant
    import mamba_ssm  # noqa: F401
except ImportError:
    stub = types.ModuleType("mamba_ssm")

    class _Mamba(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("mamba_ssm is not installed")

    stub.Mamba = _Mamba
    sys.modules["mamba_ssm"] = stub

from datasets import Audio, load_dataset  # noqa: E402
from pesq import pesq  # noqa: E402
from pystoi import stoi  # noqa: E402

from lightning_modules.lightning_module import UltraSpectrogramLightningModelUnet  # noqa: E402
from utils.edge_fix import run_fixed  # noqa: E402

SR = 16000
METRICS = ("si_sdr", "pesq_wb", "pesq_nb", "stoi")


def si_sdr(est, ref):
    ref, est = ref - ref.mean(), est - est.mean()
    target = np.dot(est, ref) / np.dot(ref, ref) * ref
    return 10 * np.log10(np.sum(target ** 2) / np.sum((est - target) ** 2))


def score(est, ref):
    est = np.nan_to_num(est)
    out = {"si_sdr": si_sdr(est, ref), "stoi": stoi(ref, est, SR, extended=False)}
    for mode in ("wb", "nb"):
        try:
            out[f"pesq_{mode}"] = pesq(SR, ref, est, mode)
        except Exception:
            out[f"pesq_{mode}"] = float("nan")
    return out


def decode(cell):
    audio, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float64")
    assert sr == SR, sr
    return audio if audio.ndim == 1 else audio.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/ultra/checkpoints/last.ckpt")
    ap.add_argument("--data", default="JacobLinCool/VoiceBank-DEMAND-16k",
                    help="Hugging Face dataset id (test split is used) or a local .parquet file")
    ap.add_argument("--out", default="vbd_edge_fix.csv")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N clips (0 = all)")
    args = ap.parse_args()

    if args.data.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=args.data, split="train")
    else:
        ds = load_dataset(args.data, split="test")
    ds = ds.cast_column("noisy", Audio(decode=False)).cast_column("clean", Audio(decode=False))
    n_clips = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    model = UltraSpectrogramLightningModelUnet.load_from_checkpoint(args.ckpt, map_location="cpu").eval()
    rows = []
    for i in range(n_clips):
        noisy, clean = decode(ds[i]["noisy"]), decode(ds[i]["clean"])
        length = min(len(noisy), len(clean))
        noisy, clean = noisy[:length], clean[:length]
        x = torch.tensor(noisy, dtype=torch.float32)
        with torch.no_grad():
            original = model.run(x.unsqueeze(0)).reshape(-1).numpy().astype(np.float64)[:length]
        fixed = run_fixed(model, x).numpy().astype(np.float64)[:length]
        for variant, y in (("noisy", noisy), ("original", original), ("fixed", fixed)):
            rows.append({"idx": i, "variant": variant, **score(y, clean)})
        print(f"{i + 1}/{n_clips}", flush=True)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["idx", "variant", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    table = {v: np.array([[r[m] for m in METRICS] for r in rows if r["variant"] == v]) for v in ("noisy", "original", "fixed")}
    print(f"\n{'variant':10s}" + "".join(f"{m:>10s}" for m in METRICS))
    for v, t in table.items():
        print(f"{v:10s}" + "".join(f"{x:10.3f}" for x in np.nanmean(t, axis=0)))
    for a, b in (("fixed", "noisy"), ("fixed", "original")):
        d = table[a] - table[b]
        ci = 1.96 * np.nanstd(d, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(d), axis=0))
        print(f"{a} - {b}: " + "; ".join(f"{m} {mu:+.3f}±{c:.3f}" for m, mu, c in zip(METRICS, np.nanmean(d, axis=0), ci)))


if __name__ == "__main__":
    main()
