"""
Generate all figures for the NanoSpatialBench paper.

Creates publication-quality placeholder figures for the paper outline.
These will be replaced with actual experimental data.

Usage:
    python generate_paper_figures.py --output_dir ../figures
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def set_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ============================================================================
# Figure 1: Benchmark Overview
# ============================================================================
def generate_fig1_benchmark_overview(output_dir: str):
    """Generate the NanoSpatialBench overview figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    colors = ['#2166AC', '#D6604D', '#4DAF4A']
    titles = [
        '(A) Sketch-to-Trajectory\nTranslation',
        '(B) Anatomical Spatial\nReasoning',
        '(C) Closed-Loop\nCorrection'
    ]
    descriptions = [
        'Input: 2D Sketch\n→ Output: 3D Waypoints',
        'Input: Scene + Query\n→ Output: Spatial Answer',
        'Input: Disturbance\n→ Output: Corrected Path',
    ]

    for ax, color, title, desc in zip(axes, colors, titles, descriptions):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')

        # Task box
        rect = patches.FancyBboxPatch((0.5, 0.5), 9, 7, boxstyle="round,pad=0.3",
                                       linewidth=2.5, edgecolor=color,
                                       facecolor=(*plt.cm.colors.to_rgba(color)[:3], 0.08))
        ax.add_patch(rect)

        ax.text(5, 6.5, title, ha='center', va='center', fontsize=12,
                fontweight='bold', color=color)

        # Icon area
        if 'Sketch' in title:
            # Draw mini sketch
            t = np.linspace(1.5, 8.5, 50)
            y_sketch = 3.5 + 0.8 * np.sin(t) + np.random.randn(50) * 0.15
            ax.plot(t, y_sketch, 'b--', alpha=0.5, linewidth=1.5, label='Sketch')
            y_smooth = 3.5 + 0.8 * np.sin(t)
            ax.plot(t, y_smooth, 'g-', linewidth=2, label='3D Output')
            ax.legend(loc='lower right', fontsize=8)
        elif 'Spatial' in title:
            # Draw scene with query
            circle = plt.Circle((5, 3.5), 1.5, fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
            ax.plot(3, 4, 'r^', markersize=12)
            ax.text(5, 3.5, '?', fontsize=24, ha='center', va='center', color=color)
            ax.text(3, 4.5, 'device', fontsize=8, ha='center')
        else:
            # Draw correction
            t = np.linspace(1.5, 8.5, 30)
            y_plan = 4 + 0.5 * np.sin(t * 0.5)
            y_actual = y_plan.copy()
            y_actual[10:15] += 1.2  # deviation
            y_corrected = y_plan.copy()
            y_corrected[12:] = np.linspace(y_actual[12], y_plan[-1], len(t)-12)
            ax.plot(t, y_plan, 'g--', alpha=0.5, linewidth=1.5, label='Planned')
            ax.plot(t[:15], y_actual[:15], 'r-', linewidth=2, label='Deviated')
            ax.plot(t[12:], y_corrected[12:], 'b-', linewidth=2, label='Corrected')
            ax.legend(loc='lower right', fontsize=8)

        ax.text(5, 1.2, desc, ha='center', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_benchmark_overview.png'))
    plt.close()
    print("Generated: fig1_benchmark_overview.png")


# ============================================================================
# Figure 2: Task 1 Results (placeholder with synthetic data)
# ============================================================================
def generate_fig2_task1_results(output_dir: str):
    """Generate Task 1 trajectory comparison figure."""
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])

    np.random.seed(42)
    z = np.linspace(0, 500, 50)

    # Ground truth
    gt_x = 20 * (1 / (1 + np.exp(-0.02 * (z - 300))))
    gt_y = 15 * (1 / (1 + np.exp(-0.02 * (z - 350))))

    # (a) Input sketch
    ax0 = fig.add_subplot(gs[0])
    sketch_x = gt_x + np.random.randn(50) * 8
    sketch_y = gt_y + np.random.randn(50) * 8
    ax0.plot(sketch_x, z, 'b-', alpha=0.5, linewidth=1.5)
    ax0.fill_betweenx([0, 500], -80, 80, alpha=0.05, color='red')
    ax0.plot([-60]*2, [0, 500], 'r-', alpha=0.3)
    ax0.plot([60]*2, [0, 500], 'r-', alpha=0.3)
    ax0.set_title('(a) Input Sketch', fontweight='bold')
    ax0.set_xlabel('X (μm)')
    ax0.set_ylabel('Z (μm)')
    ax0.set_xlim(-80, 80)

    # (b) Ground truth
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(gt_x, z, 'g-', linewidth=2.5, label='Ground Truth')
    ax1.fill_betweenx([0, 500], -80, 80, alpha=0.05, color='red')
    ax1.plot([-60]*2, [0, 500], 'r-', alpha=0.3)
    ax1.plot([60]*2, [0, 500], 'r-', alpha=0.3)
    ax1.set_title('(b) Ground Truth', fontweight='bold')
    ax1.set_xlabel('X (μm)')
    ax1.set_xlim(-80, 80)
    ax1.legend(fontsize=8)

    # (c) Model A prediction
    ax2 = fig.add_subplot(gs[2])
    pred_a = gt_x + np.random.randn(50) * 3
    ax2.plot(pred_a, z, 'c-', linewidth=2, label='GPT-4o (CoT)')
    ax2.plot(gt_x, z, 'g--', linewidth=1, alpha=0.5, label='GT')
    ax2.fill_betweenx([0, 500], -80, 80, alpha=0.05, color='red')
    ax2.set_title('(c) GPT-4o (CoT)', fontweight='bold')
    ax2.set_xlabel('X (μm)')
    ax2.set_xlim(-80, 80)
    ax2.legend(fontsize=8)

    # (d) Model B prediction
    ax3 = fig.add_subplot(gs[3])
    pred_b = gt_x + np.random.randn(50) * 1.5
    ax3.plot(pred_b, z, 'm-', linewidth=2, label='LLaMA-FT')
    ax3.plot(gt_x, z, 'g--', linewidth=1, alpha=0.5, label='GT')
    ax3.fill_betweenx([0, 500], -80, 80, alpha=0.05, color='red')
    ax3.set_title('(d) Fine-Tuned', fontweight='bold')
    ax3.set_xlabel('X (μm)')
    ax3.set_xlim(-80, 80)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_task1_results.png'))
    plt.close()
    print("Generated: fig2_task1_results.png")


# ============================================================================
# Figure 3: Closed-Loop Results (placeholder)
# ============================================================================
def generate_fig3_closedloop_results(output_dir: str):
    """Generate closed-loop correction performance figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    np.random.seed(42)

    # (a) Correction accuracy vs disturbance magnitude
    ax = axes[0]
    dist_mag = np.linspace(5, 50, 10)
    models = {
        'GPT-4o': 15 + 0.8 * dist_mag + np.random.randn(10) * 2,
        'Claude 3.5': 12 + 0.7 * dist_mag + np.random.randn(10) * 2,
        'LLaMA-FT': 8 + 0.5 * dist_mag + np.random.randn(10) * 1.5,
        'MPC': 5 + 0.3 * dist_mag + np.random.randn(10) * 1,
    }
    for name, rmse in models.items():
        ax.plot(dist_mag, rmse, 'o-', label=name, markersize=4)
    ax.set_xlabel('Disturbance Magnitude (μm)')
    ax.set_ylabel('Correction RMSE (μm)')
    ax.set_title('(a) Accuracy vs. Disturbance', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Latency distribution
    ax = axes[1]
    latencies = {
        'GPT-4o': np.random.lognormal(7.5, 0.3, 100),
        'Claude 3.5': np.random.lognormal(7.3, 0.35, 100),
        'Gemini 1.5': np.random.lognormal(7.4, 0.4, 100),
        'LLaMA-FT': np.random.lognormal(6.5, 0.25, 100),
    }
    bp = ax.boxplot([v for v in latencies.values()],
                     labels=list(latencies.keys()),
                     patch_artist=True)
    colors_bp = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('(b) Response Latency', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Safety score under perturbation
    ax = axes[2]
    perturb = np.linspace(0, 50, 10)
    safety = {
        'GPT-4o': 0.95 - 0.008 * perturb + np.random.randn(10) * 0.02,
        'Claude 3.5': 0.96 - 0.006 * perturb + np.random.randn(10) * 0.015,
        'LLaMA-FT': 0.97 - 0.004 * perturb + np.random.randn(10) * 0.01,
        'MPC': 0.99 - 0.002 * perturb + np.random.randn(10) * 0.005,
    }
    for name, s in safety.items():
        ax.plot(perturb, np.clip(s, 0, 1), 's-', label=name, markersize=4)
    ax.set_xlabel('Perturbation (μm)')
    ax.set_ylabel('Safety Score')
    ax.set_title('(c) Safety vs. Perturbation', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_closedloop_results.png'))
    plt.close()
    print("Generated: fig3_closedloop_results.png")


# ============================================================================
# Figure 4: Ablation Studies (placeholder)
# ============================================================================
def generate_fig4_ablation(output_dir: str):
    """Generate ablation study figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)

    # (a) Prompt strategy comparison
    ax = axes[0, 0]
    strategies = ['Zero-Shot', 'Few-Shot\n(k=1)', 'Few-Shot\n(k=3)', 'Few-Shot\n(k=5)', 'CoT']
    rmse_vals = [45, 35, 28, 26, 22]
    rmse_err = [5, 4, 3, 3, 2.5]
    bars = ax.bar(strategies, rmse_vals, yerr=rmse_err, capsize=4,
                  color=['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#2166AC'],
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Trajectory RMSE (μm)')
    ax.set_title('(a) Prompt Strategy Impact', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Fine-tuning data scaling
    ax = axes[0, 1]
    data_sizes = [100, 500, 1000, 5000, 10000, 50000]
    rmse_ft = [40, 30, 22, 16, 13, 11]
    rmse_ft_err = [4, 3, 2.5, 2, 1.5, 1]
    ax.errorbar(data_sizes, rmse_ft, yerr=rmse_ft_err, fmt='o-', color='#D6604D',
                capsize=4, linewidth=2, markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Trajectory RMSE (μm)')
    ax.set_title('(b) Fine-Tuning Data Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (c) Sketch imprecision robustness
    ax = axes[1, 0]
    noise_levels = [0.5, 1, 2, 3, 5]
    models_noise = {
        'GPT-4o (CoT)': [20, 22, 28, 35, 48],
        'Claude 3.5 (CoT)': [18, 20, 25, 32, 44],
        'LLaMA-FT': [14, 15, 18, 22, 30],
    }
    for name, vals in models_noise.items():
        ax.plot(noise_levels, vals, 'o-', label=name, markersize=5)
    ax.set_xlabel('Sketch Noise Level (σ μm)')
    ax.set_ylabel('Trajectory RMSE (μm)')
    ax.set_title('(c) Robustness to Sketch Noise', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Vision vs text-only
    ax = axes[1, 1]
    model_names = ['GPT-4o', 'Claude 3.5', 'LLaMA 90B']
    vision_acc = [72, 75, 65]
    text_acc = [48, 52, 42]
    x_pos = np.arange(len(model_names))
    width = 0.35
    ax.bar(x_pos - width/2, vision_acc, width, label='Vision', color='#2166AC', alpha=0.7)
    ax.bar(x_pos + width/2, text_acc, width, label='Text-Only', color='#D6604D', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Task 2 Accuracy (%)')
    ax.set_title('(d) Vision vs. Text-Only', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_ablation.png'))
    plt.close()
    print("Generated: fig4_ablation.png")


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output_dir", type=str, default="../figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_style()

    generate_fig1_benchmark_overview(args.output_dir)
    generate_fig2_task1_results(args.output_dir)
    generate_fig3_closedloop_results(args.output_dir)
    generate_fig4_ablation(args.output_dir)

    print(f"\nAll figures generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
