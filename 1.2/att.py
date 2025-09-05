# profile_attention_cpu_gpu_savefig.py
# Satisfies: FLOPs, memory, time; L in {10,100,1k,10k}; SEM error bars; CPU & GPU; save plots.

import math, time, gc, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tracemalloc
from dataclasses import dataclass

# ---------------- basic config ----------------
D_MODEL   = 128
N_HEADS   = 8
BATCH     = 1
DTYPE     = torch.float32
L_LIST    = [10, 100, 1_000, 10_000]   # reduce 10_000 if your machine OOMs
N_TRIALS  = 8                           # averaging for smaller SEM
WARMUP    = 2                           # throwaway forward passes for timing stability

# -------------- FLOPs (analytical) --------------
def flops_self_attention(L, d_model, n_heads):
    # Multiply+add counted as 2 FLOPs; common estimate:
    # total ≈ 8 L d^2 + 4 L^2 d + 5 H L^2
    d = d_model
    return int(8*L*d*d + 4*(L**2)*d + 5*n_heads*(L**2))

# -------------- helpers --------------
@dataclass
class Stat:
    mean: float
    sem: float

def mean_sem(xs):
    xs = np.array(xs, dtype=np.float64)
    mean = float(xs.mean())
    sem  = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return Stat(mean, sem)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def measure_once_cpu(mha, L):
    x = torch.randn(BATCH, L, D_MODEL, dtype=DTYPE)
    tracemalloc.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        y, _ = mha(x, x, x)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (t1 - t0), float(peak)

def measure_once_gpu(mha, L, device):
    x = torch.randn(BATCH, L, D_MODEL, dtype=DTYPE, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        y, _ = mha(x, x, x)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    peak_bytes = float(torch.cuda.max_memory_allocated(device))
    return (t1 - t0), peak_bytes

def run_trials(device):
    using_gpu = (device.type == "cuda")
    mha = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True).to(device=device, dtype=DTYPE).eval()

    # warmup
    with torch.no_grad():
        Lw = 64
        xw = torch.randn(BATCH, Lw, D_MODEL, dtype=DTYPE, device=device)
        for _ in range(WARMUP):
            _ = mha(xw, xw, xw)

    results = {}
    for L in L_LIST:
        times, mems, flps = [], [], []
        ok = True
        for _ in range(N_TRIALS):
            try:
                clear_memory()
                if using_gpu:
                    t, m = measure_once_gpu(mha, L, device)
                else:
                    t, m = measure_once_cpu(mha, L)
                times.append(t); mems.append(m)
                flps.append(flops_self_attention(L, D_MODEL, N_HEADS))
            except RuntimeError as e:
                # Likely OOM for this L on this device
                sys.stderr.write(f"[WARN] {device} L={L}: {repr(e)}\n")
                ok = False
                break
            except Exception as e:
                sys.stderr.write(f"[ERROR] {device} L={L}: {repr(e)}\n")
                traceback.print_exc()
                ok = False
                break

        if ok and times:
            results[L] = {
                "time":  mean_sem(times),
                "mem":   mean_sem(mems),
                "flops": mean_sem(flps),  # deterministic per L (SEM≈0)
            }
        else:
            results[L] = {
                "time":  Stat(float("nan"), float("nan")),
                "mem":   Stat(float("nan"), float("nan")),
                "flops": Stat(float("nan"), float("nan")),
            }
    return results

def plot_and_save(lengths, vals, errs, title, ylabel, fname):
    plt.figure()
    plt.errorbar(lengths, vals, yerr=errs, fmt='o-', capsize=4)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Sequence length (L)'); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    print(f"Saved: {fname}")

def summarize_print(device_name, results):
    for L in results:
        t = results[L]["time"]; m = results[L]["mem"]; f = results[L]["flops"]
        if not math.isnan(t.mean):
            print(f"[{device_name}] L={L:5d} | time={t.mean:.6f}s ±{t.sem:.6f} | "
                  f"peak_mem={m.mean/1e6:.2f}MB ±{m.sem/1e6:.2f} | FLOPs≈{f.mean/1e9:.3f} GF")

# -------------- main --------------
def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        dev_name = dev.type.upper()
        print(f"\n=== Running on {dev_name} ===")
        res = run_trials(dev)
        summarize_print(dev_name, res)

        # vectors for plotting (skip NaNs if any L was OOM)
        Ls      = [L for L in L_LIST if not math.isnan(res[L]["time"].mean)]
        t_means = [res[L]["time"].mean  for L in Ls]
        t_sems  = [res[L]["time"].sem   for L in Ls]
        m_means = [res[L]["mem"].mean   for L in Ls]
        m_sems  = [res[L]["mem"].sem    for L in Ls]
        f_means = [res[L]["flops"].mean for L in Ls]
        f_sems  = [res[L]["flops"].sem  for L in Ls]  # ~0

        # save figures locally
        plot_and_save(Ls, t_means, t_sems, f"Wall Time vs L ({dev_name}, MHA)", "Seconds",
                      f"time_vs_L_{dev.type}.png")
        plot_and_save(Ls, m_means, m_sems, f"Peak Memory vs L ({dev_name}, MHA)", "Bytes",
                      f"memory_vs_L_{dev.type}.png")
        plot_and_save(Ls, f_means, f_sems, f"FLOPs (Analytical) vs L ({dev_name})", "FLOPs",
                      f"flops_vs_L_{dev.type}.png")

    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
