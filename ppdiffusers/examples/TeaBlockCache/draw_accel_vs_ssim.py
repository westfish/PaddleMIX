#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed-up × coco1k-ssim 可视化
- 同模型不同参数点，用相同颜色并连线
- 无 X11 时自动转存为 PNG
依赖：python3, matplotlib
安装：pip install matplotlib
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# ---------- 数据 ----------
records = [
    ("tgate",                     1.20, 0.740),
    ("pab",                       1.57, 0.8516),
    ("blockdance",                     2.10, 0.858),
    ("teacache",                  1.73, 0.797),
    ("firstblock_taylorseer 0.07", 2.03, 0.917),
    ("firstblock_taylorseer 0.14", 3.26, 0.799),
    ("teablockcache 0.0", 0.93, 1.0),
    ("teablockcache 0.3", 1.34, 0.9833),
    ("teablockcache 0.9", 1.66, 0.9385),
    ("teablockcache_taylor 2", 3.99, 0.7663),
    ("teablockcache_taylor 1", 2.98, 0.8665),
    ("sortblockcache", 1.66, 0.95176),
]
# --------------------------

# 无图形界面时切换到非交互后端
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# 按模型分组
groups = {}
for full_name, spd, ssim in records:
    base = full_name.split()[0]          # 取空格前部分作为“模型名”
    groups.setdefault(base, []).append((spd, ssim, full_name))

fig, ax = plt.subplots(figsize=(7, 4))

for base, pts in groups.items():
    pts.sort(key=lambda x: x[0])         # 按加速比升序，便于画线
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    scatter = ax.scatter(xs, ys, label=base)     # 散点
    if len(pts) > 1:
        ax.plot(xs, ys, color=scatter.get_facecolors()[0])  # 连线

    for x, y, lbl in pts:               # 给每点加完整标签
        ax.text(x, y, lbl, fontsize=8, ha="right", va="bottom")

ax.set_xlabel("Speed-up (×)")
ax.set_ylabel("coco1k-ssim (↑)")
ax.set_title("Speed-up vs. Image Quality (SSIM)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Model", fontsize=9)
plt.tight_layout()

# 有 / 无 GUI 的两种输出方式
if matplotlib.get_backend() == "Agg":
    out_file = "accel_vs_ssim.png"
    plt.savefig(out_file, dpi=150)
    print(f"无图形界面：图已保存为 {out_file}")
else:
    plt.show()