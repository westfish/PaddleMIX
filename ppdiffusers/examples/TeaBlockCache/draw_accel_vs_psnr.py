#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed-up × coco1k-PSNR 可视化
- 同模型不同参数点，用相同颜色并连线
- 无 X11 时自动转存为 PNG
依赖：python3, matplotlib
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# ---------- 数据 ----------
# 说明：第三列请填入 *实际* coco1k-PSNR（数值越高越好）
records = [
    ("tgate",                     1.20, 20.11),
    ("pab",                       1.57, 24.67),
    ("blockdance",                2.10, 24.65),
    ("teacache",                  1.73, 21.68),
    ("firstblock_taylorseer 0.07", 2.03, 27.92),
    ("firstblock_taylorseer 0.14", 3.26, 21.74),
    # ("teablockcache 0.0",    0.93, 100.00),
    ("teablockcache 0.3",    1.34,  37.17),
    ("teablockcache 0.9",    1.66,  29.81),
    ("teablockcache_taylor 2",    3.99,  20.46),
    ("teablockcache_taylor 1",    2.98,  24.47),
    ("teablockcache_taylor 0.5",  2.13,  29.49),
    ("sortblockcache",            1.66,  31.1875),
]
# --------------------------

# 无图形界面时切换到非交互后端
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# 按模型分组
groups = {}
for full_name, spd, psnr in records:
    base = full_name.split()[0]          # 取空格前部分作为“模型名”
    groups.setdefault(base, []).append((spd, psnr, full_name))

fig, ax = plt.subplots(figsize=(7, 4))

for base, pts in groups.items():
    pts.sort(key=lambda x: x[0])         # 按加速比升序，便于画线
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    scatter = ax.scatter(xs, ys, label=base)     # 散点
    if len(pts) > 1:
        ax.plot(xs, ys, color=scatter.get_facecolors()[0])  # 连线

    for x, y, lbl in pts:                         # 给每点加完整标签
        ax.text(x, y, lbl, fontsize=8, ha="right", va="bottom")

ax.set_xlabel("Speed-up (×)")
ax.set_ylabel("coco1k-PSNR (↑)")
ax.set_title("Speed-up vs. Image Quality (PSNR)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Model", fontsize=9)
plt.tight_layout()

# 有 / 无 GUI 的两种输出方式
if matplotlib.get_backend() == "Agg":
    out_file = "accel_vs_psnr.png"
    plt.savefig(out_file, dpi=150)
    print(f"无图形界面：图已保存为 {out_file}")
else:
    plt.show()