"""
Generate synthetic benchmark data for NanoSpatialBench.

This script creates sample data for all three tasks using parametric models
of anatomical structures and sketch-based trajectory planning.

Usage:
    python generate_sample_data.py --task all --num_samples 100 --output_dir ../data
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def generate_vessel_geometry(seed: int = 42) -> dict:
    """Generate a random parameterized vessel tree geometry."""
    rng = np.random.RandomState(seed)

    # Main vessel: curved tube along z-axis
    length = rng.uniform(500, 2000)  # micrometers
    radius = rng.uniform(50, 200)
    curvature = rng.uniform(0, 0.005)  # slight bend

    # Optional bifurcation
    has_bifurcation = rng.random() > 0.5
    bifurcation_z = rng.uniform(0.3, 0.7) * length if has_bifurcation else None

    # Optional stenosis (narrowing)
    has_stenosis = rng.random() > 0.6
    stenosis_z = rng.uniform(0.2, 0.8) * length if has_stenosis else None
    stenosis_factor = rng.uniform(0.3, 0.7) if has_stenosis else None

    # Tumor target
    tumor_z = rng.uniform(0.6, 0.95) * length
    tumor_offset = rng.uniform(0, radius * 0.6)
    tumor_angle = rng.uniform(0, 2 * np.pi)
    tumor_x = tumor_offset * np.cos(tumor_angle)
    tumor_y = tumor_offset * np.sin(tumor_angle)
    tumor_radius = rng.uniform(20, 80)

    return {
        "length": length,
        "radius": radius,
        "curvature": curvature,
        "has_bifurcation": has_bifurcation,
        "bifurcation_z": bifurcation_z,
        "has_stenosis": has_stenosis,
        "stenosis_z": stenosis_z,
        "stenosis_factor": stenosis_factor,
        "tumor_position": [float(tumor_x), float(tumor_y), float(tumor_z)],
        "tumor_radius": tumor_radius,
    }


def generate_ground_truth_trajectory(geometry: dict, n_waypoints: int = 20) -> list:
    """Generate a smooth ground-truth trajectory through the vessel to the tumor."""
    z_vals = np.linspace(0, geometry["tumor_position"][2], n_waypoints)

    # Smooth path towards tumor
    t = z_vals / geometry["tumor_position"][2]  # normalized [0,1]

    # Sigmoid approach to tumor in x, y
    sigmoid = 1 / (1 + np.exp(-10 * (t - 0.7)))
    x_vals = sigmoid * geometry["tumor_position"][0]
    y_vals = sigmoid * geometry["tumor_position"][1]

    # Add vessel curvature
    x_vals += geometry["curvature"] * z_vals * 100

    # Velocity profile: slower near target
    velocities = 100 * (1 - 0.7 * sigmoid)  # um/s

    waypoints = []
    for i in range(n_waypoints):
        waypoints.append([float(x_vals[i]), float(y_vals[i]), float(z_vals[i])])

    return waypoints


def generate_sketch_description(geometry: dict) -> str:
    """Generate a text description of what a sketch would show."""
    desc = "A longitudinal view of a blood vessel"

    if geometry["has_stenosis"]:
        desc += " with a narrowing (stenosis) at the midpoint"
    if geometry["has_bifurcation"]:
        desc += " with a bifurcation (branching point)"

    desc += ". A tumor mass is indicated near the distal end."
    desc += " An arrow traces the intended nano-device path from the vessel entrance to the tumor."

    return desc


def generate_task1_data(num_samples: int, output_dir: str):
    """Generate Task 1: Sketch-to-3D Trajectory Translation data."""
    data = []

    for i in range(num_samples):
        geometry = generate_vessel_geometry(seed=i)
        gt_waypoints = generate_ground_truth_trajectory(geometry)

        instance = {
            "id": f"task1_{i:04d}",
            "sketch_path": "",  # To be filled with actual sketch images
            "sketch_description": generate_sketch_description(geometry),
            "anatomy_context": f"Blood vessel (diameter: {2*geometry['radius']:.0f} um, "
                              f"length: {geometry['length']:.0f} um). "
                              f"Tumor at approximately ({geometry['tumor_position'][0]:.0f}, "
                              f"{geometry['tumor_position'][1]:.0f}, "
                              f"{geometry['tumor_position'][2]:.0f}) um.",
            "scale_info": f"1 pixel = 10 micrometers. Vessel diameter = {2*geometry['radius']:.0f} um.",
            "ground_truth_waypoints": gt_waypoints,
            "boundaries": {
                "center": [0.0, 0.0],
                "radius": geometry["radius"],
            },
            "geometry": geometry,
        }
        data.append(instance)

    output_path = Path(output_dir) / "task1_data.jsonl"
    with open(output_path, "w") as f:
        for instance in data:
            # Remove non-serializable geometry for clean output
            clean = {k: v for k, v in instance.items() if k != "geometry"}
            f.write(json.dumps(clean) + "\n")

    print(f"Generated {num_samples} Task 1 instances -> {output_path}")


def generate_task2_data(num_samples: int, output_dir: str):
    """Generate Task 2: Anatomical Spatial Relationship Reasoning data."""
    rng = np.random.RandomState(42)

    query_templates = [
        ("Is the device proximal or distal to the {landmark}?",
         ["proximal", "distal"]),
        ("Is the device inside or outside the {region}?",
         ["inside", "outside"]),
        ("Is the device anterior or posterior to the {structure}?",
         ["anterior", "posterior"]),
        ("Which vessel branch is the device closest to: {branch_a} or {branch_b}?",
         ["{branch_a}", "{branch_b}"]),
    ]

    landmarks = ["bifurcation", "stenosis", "tumor margin", "vessel entrance"]
    regions = ["tumor", "vessel lumen", "stenotic region"]
    structures = ["vessel wall", "tumor core", "bifurcation point"]
    branches = [("left branch", "right branch"), ("main trunk", "side branch")]

    data = []
    for i in range(num_samples):
        geometry = generate_vessel_geometry(seed=i + 1000)

        # Random device position
        dev_z = rng.uniform(0, geometry["length"])
        dev_r = rng.uniform(0, geometry["radius"] * 1.2)
        dev_angle = rng.uniform(0, 2 * np.pi)
        dev_pos = [
            float(dev_r * np.cos(dev_angle)),
            float(dev_r * np.sin(dev_angle)),
            float(dev_z),
        ]

        # Select query type
        q_idx = rng.randint(len(query_templates))
        template, answer_options = query_templates[q_idx]

        if q_idx == 0:
            landmark = rng.choice(landmarks)
            query = template.format(landmark=landmark)
            ref_z = geometry.get("bifurcation_z") or geometry["length"] * 0.5
            answer = "proximal" if dev_z < ref_z else "distal"
        elif q_idx == 1:
            region = rng.choice(regions)
            query = template.format(region=region)
            if region == "tumor":
                dist = np.sqrt(sum((d - t)**2 for d, t in zip(dev_pos, geometry["tumor_position"])))
                answer = "inside" if dist < geometry["tumor_radius"] else "outside"
            elif region == "vessel lumen":
                dist_xy = np.sqrt(dev_pos[0]**2 + dev_pos[1]**2)
                answer = "inside" if dist_xy < geometry["radius"] else "outside"
            else:
                answer = rng.choice(answer_options)
        elif q_idx == 2:
            structure = rng.choice(structures)
            query = template.format(structure=structure)
            answer = "anterior" if dev_pos[1] > 0 else "posterior"
        else:
            ba, bb = branches[rng.randint(len(branches))]
            query = template.format(branch_a=ba, branch_b=bb)
            answer = ba if dev_pos[0] < 0 else bb

        scene = (f"Blood vessel ({2*geometry['radius']:.0f} um diameter) with a tumor "
                f"at ({geometry['tumor_position'][0]:.0f}, {geometry['tumor_position'][1]:.0f}, "
                f"{geometry['tumor_position'][2]:.0f}) um.")

        instance = {
            "id": f"task2_{i:04d}",
            "scene_description": scene,
            "device_position": dev_pos,
            "query": query,
            "ground_truth_answer": answer,
        }
        data.append(instance)

    output_path = Path(output_dir) / "task2_data.jsonl"
    with open(output_path, "w") as f:
        for instance in data:
            f.write(json.dumps(instance) + "\n")

    print(f"Generated {num_samples} Task 2 instances -> {output_path}")


def generate_task3_data(num_samples: int, output_dir: str):
    """Generate Task 3: Closed-Loop Trajectory Correction data."""
    rng = np.random.RandomState(42)

    disturbance_types = [
        "Sudden flow increase pushing device {direction} by {magnitude} um",
        "Tissue contraction narrowing vessel by {percent}%",
        "Obstacle detected: {obstacle} at ({ox}, {oy}, {oz}) um",
        "Device orientation shifted {degrees} degrees due to magnetic interference",
    ]

    data = []
    for i in range(num_samples):
        geometry = generate_vessel_geometry(seed=i + 2000)
        gt_trajectory = generate_ground_truth_trajectory(geometry)

        # Simulate device at some point along trajectory with perturbation
        progress = rng.uniform(0.2, 0.8)
        wp_idx = int(progress * len(gt_trajectory))

        # Current position = planned + perturbation
        perturbation = rng.randn(3) * 20  # 20 um std perturbation
        current_pos = [
            gt_trajectory[wp_idx][0] + perturbation[0],
            gt_trajectory[wp_idx][1] + perturbation[1],
            gt_trajectory[wp_idx][2] + perturbation[2],
        ]

        # Disturbance description
        d_type = rng.randint(len(disturbance_types))
        if d_type == 0:
            dist_desc = disturbance_types[0].format(
                direction=rng.choice(["left", "right", "upward"]),
                magnitude=f"{rng.uniform(10, 50):.0f}"
            )
        elif d_type == 1:
            dist_desc = disturbance_types[1].format(percent=f"{rng.uniform(10, 40):.0f}")
        elif d_type == 2:
            ox = rng.uniform(-50, 50)
            oy = rng.uniform(-50, 50)
            oz = current_pos[2] + rng.uniform(20, 100)
            dist_desc = disturbance_types[2].format(
                obstacle=rng.choice(["blood clot", "tissue debris", "cell cluster"]),
                ox=f"{ox:.0f}", oy=f"{oy:.0f}", oz=f"{oz:.0f}"
            )
        else:
            dist_desc = disturbance_types[3].format(degrees=f"{rng.uniform(5, 45):.0f}")

        # Ground truth correction: remaining trajectory adjusted from current position
        gt_correction = gt_trajectory[wp_idx:]
        if len(gt_correction) < 2:
            gt_correction = gt_trajectory[-2:]

        instance = {
            "id": f"task3_{i:04d}",
            "original_trajectory": gt_trajectory,
            "current_position": [float(c) for c in current_pos],
            "target": gt_trajectory[-1],
            "disturbance": dist_desc,
            "constraints": f"Vessel radius: {geometry['radius']:.0f} um. "
                          f"Minimum wall clearance: 10 um.",
            "ground_truth_correction": gt_correction,
        }
        data.append(instance)

    output_path = Path(output_dir) / "task3_data.jsonl"
    with open(output_path, "w") as f:
        for instance in data:
            f.write(json.dumps(instance) + "\n")

    print(f"Generated {num_samples} Task 3 instances -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate NanoSpatialBench data")
    parser.add_argument("--task", type=str, default="all",
                        choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="../data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    generators = {
        "task1": generate_task1_data,
        "task2": generate_task2_data,
        "task3": generate_task3_data,
    }

    for task in tasks:
        generators[task](args.num_samples, args.output_dir)

    print(f"\nAll data generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
