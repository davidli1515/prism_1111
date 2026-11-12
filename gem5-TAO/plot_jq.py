import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# 1. 数据准备
data = [
    [4, "gemm_tiling_knm_optimized",   512, 768, 768, 512, 176, 3.748, 0.198, 3.197, 1.42,  8.564, 70.53],
    [4, "gemm_spm_vl4_knm_optimized",  512, 768, 768, 256, 768, 3.722, 0.038, 3.243, 0.349, 7.36,  82.06],
    [4, "gemm_no_tiling_optimized",    512, 768, 768, 0,   0,   3.722, 0.224, 4.138, 0,     8.083, 74.72],
    [8, "gemm_tiling_knm_optimized",   512, 768, 768, 512, 96,  3.752, 0.184, 1.949, 1.422, 7.307, 82.66],
    [8, "gemm_spm_vl8_knm_optimized",  512, 768, 768, 256, 768, 3.723, 0.038, 2.374, 0.353, 6.493, 93.02],
    [8, "gemm_no_tiling_optimized",    512, 768, 768, 0,   0,   3.721, 0.223, 4.049, 0,     7.993, 75.57],
    [4, "gemm_tiling_knm_optimized",   512, 576, 768, 512, 176, 3.75,  0.148, 2.405, 1.054, 7.357, 61.57],
    [4, "gemm_spm_vl4_knm_optimized",  512, 576, 768, 256, 768, 3.722, 0.029, 2.431, 0.265, 6.453, 70.2],
    [4, "gemm_no_tiling_optimized",    512, 576, 768, 0,   0,   3.719, 0.165, 3.053, 0,     6.936, 65.31],
    [8, "gemm_tiling_knm_optimized",   512, 576, 768, 512, 96,  3.754, 0.137, 1.461, 0.959, 6.311, 71.77],
    [8, "gemm_spm_vl8_knm_optimized",  512, 576, 768, 256, 768, 3.722, 0.028, 1.781, 0.263, 5.798, 78.13],
    [8, "gemm_no_tiling_optimized",    512, 576, 768, 0,   0,   3.719, 0.163, 2.994, 0,     6.876, 65.88],
    [4, "gemm_tiling_knm_optimized",   512, 768, 512, 512, 256, 2.496, 0.145, 2.147, 0.464, 5.251, 76.68],
    [4, "gemm_spm_vl4_knm_optimized",  512, 768, 512, 256, 512, 2.491, 0.026, 2.172, 0.344, 5.039, 79.9],
    [4, "gemm_no_tiling_optimized",    512, 768, 512, 0,   0,   2.491, 0.146, 2.819, 0,     5.456, 73.8],
    [8, "gemm_tiling_knm_optimized",   512, 768, 512, 512, 128, 2.501, 0.128, 1.301, 0.698, 4.628, 87.0],
    [8, "gemm_spm_vl8_knm_optimized",  512, 768, 512, 256, 512, 2.491, 0.025, 1.593, 0.337, 4.451, 90.46],
    [8, "gemm_no_tiling_optimized",    512, 768, 512, 0,   0,   2.491, 0.145, 2.752, 0,     5.389, 74.72],
    [4, "gemm_tiling_knm_optimized",   256, 288, 2048, 256, 256, 4.966, 0.17,  1.6,   0.345, 7.081, 42.65],
    [4, "gemm_spm_vl4_knm_optimized",  256, 288, 2048, 128, 1024, 4.938, 0.039, 1.617, 0.095, 6.694, 45.11],
    [4, "gemm_no_tiling_optimized",    256, 288, 2048, 0,   0,    4.942, 0.221, 2.057, 0,     7.22,  41.83],
    [8, "gemm_tiling_knm_optimized",   256, 384, 2048, 256, 128,  5.008, 0.224, 1.215, 0.498, 6.945, 57.97],
    [8, "gemm_spm_vl8_knm_optimized",  256, 384, 2048, 128, 1024, 4.94,  0.05,  1.581, 0.123, 6.699, 60.1],
    [8, "gemm_no_tiling_optimized",    256, 384, 2048, 0,   0,    4.944, 0.299, 2.678, 0,     7.921, 50.83],
    [4, "gemm_tiling_knm_optimized",   512, 576, 2048, 256, 256,  9.928, 0.385, 6.402, 1.781, 18.496, 65.31],
    [4, "gemm_spm_vl4_knm_optimized",  512, 576, 2048, 256, 1024, 9.875, 0.075, 6.47,  0.411, 16.843, 71.72],
    [4, "gemm_no_tiling_optimized",    512, 576, 2048, 0,   0,    9.883, 0.442, 8.239, 0,     18.564, 65.07],
    [8, "gemm_tiling_knm_optimized",   512, 576, 2048, 256, 128,  10.006, 0.359, 3.615, 1.898, 15.878, 76.08],
    [8, "gemm_spm_vl8_knm_optimized",  512, 576, 2048, 256, 1024, 9.876, 0.075, 4.742, 0.382, 15.082, 80.09],
    [8, "gemm_no_tiling_optimized",    512, 576, 2048, 0,   0,    9.883, 0.439, 8.052, 0,     18.374, 65.74],
]

df = pd.DataFrame(data, columns=["VL", "Type", "M", "N", "K", "MC", "KC",
                                  "t_packA", "t_packB", "t_kernel", "t_accum", "t_total", "GFLOPS"])

def type_to_binary(t):
    if "no_tiling" in t:
        return "ACL"
    elif "spm" in t:
        return "ACL+Tiling+CacheFlex"
    elif "tiling" in t:
        return "ACL+Tiling"
    return t

df["Binary"] = df["Type"].map(type_to_binary)

# 2. 绘图配置
config_labels = [
    "M=512, K=768, N=768",
    "M=512, K=768, N=576",
    "M=512, K=512, N=768",
    "M=256, K=2048, N=256",
    "M=512, K=2048, N=576"
]

binaries = ["ACL", "ACL+Tiling", "ACL+Tiling+CacheFlex"]
display_labels = ["ACL", "ACL\n+Tiling", "ACL+Tiling\n+CacheFlex"]
latencies = ["t_packB", "t_accum", "t_kernel"]
latency_labels = {"t_packB": "Pack B", "t_accum": "C Accumulate", "t_kernel": "Computing Kernel"}

# 颜色已修复
colors = {
    "t_packB": "#4CAF50",  # 绿色
    "t_accum": "#FF9800",  # 橙色
    "t_kernel": "#2196F3"  # 蓝色
}

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(5, 1, figsize=(18, 34))
fig.patch.set_facecolor('white')

for idx, label in enumerate(config_labels):
    ax = axes[idx]

    if idx == 0:
        subset_all = df[(df["M"] == 512) & (df["K"] == 768) & (df["N"] == 768)]
    elif idx == 1:
        subset_all = df[(df["M"] == 512) & (df["K"] == 768) & (df["N"] == 576)]
    elif idx == 2:
        subset_all = df[(df["M"] == 512) & (df["K"] == 512) & (df["N"] == 768)]
    elif idx == 3:
        subset_all = df[(df["M"] == 256) & (df["K"] == 2048)]
    elif idx == 4:
        subset_all = df[(df["M"] == 512) & (df["K"] == 2048) & (df["N"] == 576)]

    x_vl4 = np.arange(len(binaries))
    x_vl8 = np.arange(len(binaries)) + 4
    width = 0.8

    for i, b in enumerate(binaries):
        for x_pos, vl_val in zip([x_vl4[i], x_vl8[i]], [4, 8]):
            subset_filtered = subset_all[(subset_all["Binary"] == b) & (subset_all["VL"] == vl_val)]

            if not subset_filtered.empty:
                row = subset_filtered.iloc[0]
                bottom = 0
                for lat in latencies:
                    val = row[lat]
                    if val > 0:
                        ax.bar(x_pos, val, width, bottom=bottom,
                               color=colors[lat], alpha=0.9,
                               edgecolor="black", linewidth=0.8)
                        if val > 0.3:
                            # 字体修改：增大到 12 (原 9)
                            ax.text(x_pos, bottom + val/2, f'{val:.2f}',
                                  ha="center", va="center",
                                  color="white", fontsize=18, fontweight="bold") # <--- 修改
                        bottom += val

    # 设置 X 轴
    ax.set_xticks(np.concatenate([x_vl4, x_vl8]))
    # 字体修改：增大到 14 (原 11)
    ax.set_xticklabels(display_labels * 2, fontsize=24, fontweight='bold') # <--- 修改
    ax.axvline(x=3, color='gray', linestyle='--', linewidth=2, alpha=0.7)

    # 矢量长度标签
    ylim_top = ax.get_ylim()[1]
    # 字体修改：增大到 16 (原 14)
    ax.text(1, ylim_top * 0.95, 'Vector Length = 512b', ha='center', va='top',
           fontsize=24, fontweight='bold', # <--- 修改
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#81C784', alpha=0.8))
    
    # 字体修改：增大到 16 (原 14)
    ax.text(5, ylim_top * 0.95, 'Vector Length = 1024b', ha='center', va='top',
           fontsize=24, fontweight='bold', # <--- 修改
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF8A65', alpha=0.8))

    # 标题修改：增大到 20 (原 16)
    ax.set_title(f"Configuration {idx + 1}: {label}",
                fontsize=26, fontweight='bold', pad=20) # <--- 修改
    
    # Y轴修改：增大到 16 (原 13)
    ax.set_ylabel("Latency (ms)", fontsize=16, fontweight='bold') # <--- 修改
    
    ax.grid(axis="y", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlim(-0.8, 6.8)
    ax.set_facecolor('#F5F5F5')

legend_elements = [Patch(facecolor=colors[lat], edgecolor='black',
                        label=latency_labels[lat], alpha=0.9)
                  for lat in latencies]

# 图例修改：内容增大到 16，标题增大到 18
fig.legend(handles=legend_elements,
          title="Latency Components",
          loc="upper center",
          bbox_to_anchor=(0.5, 0.998),
          ncol=3,
          fontsize=16,             # <--- 修改
          title_fontsize=18,       # <--- 修改
          frameon=True,
          shadow=True,
          fancybox=True)

plt.tight_layout(rect=[0, 0, 1, 0.975])

output_path = "jq.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

print(f"修改完成！字号已调大，图片保存至: {output_path}")