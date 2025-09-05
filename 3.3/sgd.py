import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 准备输出目录
# ==============================
out_dir = "./outputs"
os.makedirs(out_dir, exist_ok=True)

# ==============================
# 定义目标函数
# ==============================
def quad_min_fn(v):
    return (v ** 2).sum()

def quad_max_fn(v):
    return -(v ** 2).sum()

# ==============================
# 运行一次SGD
# ==============================
def run_sgd(fn, start, steps=60, lr=0.1, momentum=0.0,
            weight_decay=0.0, maximize=False):
    xy = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([xy], lr=lr, momentum=momentum,
                          weight_decay=weight_decay, maximize=maximize)
    traj = [xy.detach().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        loss = fn(xy)
        loss.backward()
        opt.step()
        traj.append(xy.detach().numpy().copy())
    return np.stack(traj)

# ==============================
# 画等高线
# ==============================
def plot_contours(ax, fn_name="min", lim=3.0):
    xs = np.linspace(-lim, lim, 200)
    ys = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = X**2 + Y**2 if fn_name=="min" else -X**2 - Y**2
    cs = ax.contour(X, Y, Z, levels=15)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_aspect("equal")

# ==============================
# 单次实验画图并保存
# ==============================
def plot_single_run(min_or_max="min", momentum=0.0, weight_decay=0.0,
                    start=(2.5, -2.0), step_show=3, save_prefix="exp"):
    if min_or_max == "min":
        fn = quad_min_fn
        title_core = "Minimize f(x,y)=x^2+y^2"
        maximize = False
    else:
        fn = quad_max_fn
        title_core = "Maximize f(x,y)=-x^2-y^2"
        maximize = True

    traj = run_sgd(fn, start=start, steps=60,
                   lr=0.15 if momentum<0.9 else 0.12,
                   momentum=momentum, weight_decay=weight_decay,
                   maximize=maximize)

    fig, ax = plt.subplots(figsize=(6,5))
    plot_contours(ax, "min" if min_or_max=="min" else "max")
    ax.plot(traj[::step_show,0], traj[::step_show,1],
            marker="o", markersize=5, linewidth=1.2)
    suffix = "" if weight_decay==0.0 else f"_wd{weight_decay}"
    ax.set_title(f"{title_core}\n(momentum={momentum}{suffix})")

    # 保存文件
    fname = f"{save_prefix}_{min_or_max}_m{momentum}{suffix}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=160)
    plt.close(fig)
    print(f"Saved: {fpath}")

# ==============================
# 实验 (a) Minimization
# ==============================
for m in [0.0, 0.5, 0.9]:
    plot_single_run("min", momentum=m, weight_decay=0.0, save_prefix="exp_a")

# 实验 (b) Minimization with weight_decay=0.1
for m in [0.0, 0.5, 0.9]:
    plot_single_run("min", momentum=m, weight_decay=0.1, save_prefix="exp_b")

# 实验 (c) Maximization
for m in [0.0, 0.5, 0.9]:
    plot_single_run("max", momentum=m, weight_decay=0.0, save_prefix="exp_c")
