"""
Generate synthetic medical sketch images for NanoSpatialBench.

Creates 2D sketch-style images of anatomical structures (vessels, tumors)
with hand-drawn-style trajectory annotations.

Usage:
    python generate_sketch_images.py --num_sketches 50 --output_dir ../figures/sketches
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from pathlib import Path


def draw_hand_drawn_line(ax, x_points, y_points, color='black', linewidth=2,
                          noise_level=0.5):
    """Draw a line with hand-drawn style jitter."""
    # Interpolate for smoother curves
    t_fine = np.linspace(0, 1, len(x_points) * 10)
    t_orig = np.linspace(0, 1, len(x_points))

    x_fine = np.interp(t_fine, t_orig, x_points)
    y_fine = np.interp(t_fine, t_orig, y_points)

    # Add hand-drawn noise
    noise_x = np.cumsum(np.random.randn(len(x_fine)) * noise_level * 0.01)
    noise_y = np.cumsum(np.random.randn(len(y_fine)) * noise_level * 0.01)

    ax.plot(x_fine + noise_x, y_fine + noise_y, color=color, linewidth=linewidth,
            alpha=0.8, solid_capstyle='round')


def draw_vessel(ax, x_start, y_center, length, radius, has_stenosis=False,
                stenosis_pos=0.5, stenosis_factor=0.5):
    """Draw a blood vessel cross-section with optional stenosis."""
    x = np.linspace(x_start, x_start + length, 100)

    # Upper wall
    y_upper = y_center + radius * np.ones_like(x)
    y_lower = y_center - radius * np.ones_like(x)

    if has_stenosis:
        stenosis_x = x_start + stenosis_pos * length
        stenosis_width = length * 0.15
        stenosis_effect = (1 - stenosis_factor) * np.exp(-((x - stenosis_x) / stenosis_width)**2)
        y_upper -= radius * stenosis_effect
        y_lower += radius * stenosis_effect

    draw_hand_drawn_line(ax, x, y_upper, color='#CC4444', linewidth=2.5, noise_level=0.3)
    draw_hand_drawn_line(ax, x, y_lower, color='#CC4444', linewidth=2.5, noise_level=0.3)

    return x, y_upper, y_lower


def draw_tumor(ax, cx, cy, radius):
    """Draw a tumor as an irregular blob."""
    theta = np.linspace(0, 2*np.pi, 50)
    r = radius * (1 + 0.2 * np.sin(3*theta) + 0.15 * np.cos(5*theta) +
                  0.1 * np.random.randn(len(theta)))

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    ax.fill(x, y, color='#FFB3B3', alpha=0.5)
    draw_hand_drawn_line(ax, x, y, color='#CC0000', linewidth=2, noise_level=0.2)
    ax.text(cx, cy, '×', fontsize=16, ha='center', va='center', color='red', fontweight='bold')


def draw_trajectory_arrow(ax, x_points, y_points):
    """Draw a sketched trajectory arrow."""
    draw_hand_drawn_line(ax, x_points, y_points, color='#2266CC', linewidth=2.5, noise_level=0.8)

    # Arrowhead at the end
    dx = x_points[-1] - x_points[-2]
    dy = y_points[-1] - y_points[-2]
    ax.annotate('', xy=(x_points[-1], y_points[-1]),
                xytext=(x_points[-1] - dx*0.3, y_points[-1] - dy*0.3),
                arrowprops=dict(arrowstyle='->', color='#2266CC', lw=2.5))


def generate_sketch(idx: int, output_dir: str):
    """Generate one synthetic medical sketch."""
    rng = np.random.RandomState(idx)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Slight background texture (paper-like)
    ax.set_facecolor('#FAFAF5')

    # Draw vessel
    vessel_y = 3.0
    vessel_radius = 0.8 + rng.uniform(-0.2, 0.2)
    has_stenosis = rng.random() > 0.5

    draw_vessel(ax, 0.5, vessel_y, 8.5, vessel_radius,
                has_stenosis=has_stenosis,
                stenosis_pos=rng.uniform(0.3, 0.6),
                stenosis_factor=rng.uniform(0.3, 0.5))

    # Draw tumor
    tumor_x = rng.uniform(6.5, 8.5)
    tumor_y = vessel_y + rng.uniform(-vessel_radius*0.5, vessel_radius*0.5)
    tumor_r = rng.uniform(0.3, 0.6)
    draw_tumor(ax, tumor_x, tumor_y, tumor_r)

    # Draw trajectory arrow from vessel entrance to tumor
    n_points = 15
    x_traj = np.linspace(1.0, tumor_x, n_points)
    # Smooth curve towards tumor
    t = np.linspace(0, 1, n_points)
    y_traj = vessel_y + (tumor_y - vessel_y) * (1 / (1 + np.exp(-8 * (t - 0.6))))

    draw_trajectory_arrow(ax, x_traj, y_traj)

    # Add nano-device start marker
    ax.plot(1.0, vessel_y, 'o', color='#2266CC', markersize=8)
    ax.text(1.0, vessel_y - 0.4, 'start', fontsize=9, ha='center',
            color='#2266CC', style='italic')

    # Add labels
    ax.text(0.3, vessel_y + vessel_radius + 0.3, 'vessel wall', fontsize=9,
            color='#CC4444', style='italic')
    ax.text(tumor_x, tumor_y + tumor_r + 0.3, 'tumor', fontsize=9,
            ha='center', color='#CC0000', style='italic')

    if has_stenosis:
        ax.text(4.0, vessel_y - vessel_radius - 0.4, 'stenosis', fontsize=8,
                ha='center', color='#888888', style='italic')

    # Save
    output_path = Path(output_dir) / f"sketch_{idx:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#FAFAF5')
    plt.close()

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical sketches")
    parser.add_argument("--num_sketches", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="../figures/sketches")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_sketches):
        path = generate_sketch(i, args.output_dir)
        if i % 10 == 0:
            print(f"Generated {i+1}/{args.num_sketches}: {path}")

    print(f"\nGenerated {args.num_sketches} sketches in: {args.output_dir}")


if __name__ == "__main__":
    main()
