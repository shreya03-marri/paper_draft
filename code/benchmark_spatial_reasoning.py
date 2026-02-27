"""
NanoSpatialBench: Benchmark for LLM Spatial Reasoning in Nano-Device Control
=============================================================================

All models run locally on NVIDIA Jetson AGX Orin 64GB via Ollama or vLLM.
No paid API keys required.

Usage:
    # Using Ollama (recommended for quick setup):
    python benchmark_spatial_reasoning.py --task task1 --model qwen2.5-vl:7b --backend ollama

    # Using vLLM (recommended for throughput benchmarking):
    python benchmark_spatial_reasoning.py --task task1 --model Qwen/Qwen2.5-VL-7B-Instruct --backend vllm

    # Run all tasks on all models:
    python benchmark_spatial_reasoning.py --task all --model all --backend ollama

Tasks:
    task1: Sketch-to-3D Trajectory Translation
    task2: Anatomical Spatial Relationship Reasoning
    task3: Closed-Loop Trajectory Correction

Supported models (Ollama names / HuggingFace IDs):
    VLMs:
        qwen2.5-vl:7b      / Qwen/Qwen2.5-VL-7B-Instruct
        qwen2.5-vl:3b      / Qwen/Qwen2.5-VL-3B-Instruct
        llama3.2-vision:11b / meta-llama/Llama-3.2-11B-Vision-Instruct
        llava:13b           / liuhaotian/llava-v1.6-mistral-7b
        phi3.5-vision       / microsoft/Phi-3.5-vision-instruct
    Text-only baselines:
        llama3.1:8b         / meta-llama/Llama-3.1-8B-Instruct
        qwen2.5:7b          / Qwen/Qwen2.5-7B-Instruct

Requirements (Jetson AGX Orin):
    pip install numpy scipy matplotlib Pillow requests tqdm pandas
    # Then install ONE of:
    #   Ollama:  curl -fsSL https://ollama.com/install.sh | sh
    #   vLLM:    pip install vllm  (with JetPack 6.x)
"""

import argparse
import json
import os
import time
import base64
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================
OLLAMA_MODELS = {
    "qwen2.5-vl:7b": {"vision": True, "params": "7B"},
    "qwen2.5-vl:3b": {"vision": True, "params": "3B"},
    "llama3.2-vision:11b": {"vision": True, "params": "11B"},
    "llava:13b": {"vision": True, "params": "13B"},
    "phi3.5-vision": {"vision": True, "params": "4.2B"},
    "llama3.1:8b": {"vision": False, "params": "8B"},
    "qwen2.5:7b": {"vision": False, "params": "7B"},
}

# Mapping for HuggingFace model IDs (used with vLLM backend)
HF_MODELS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": {"vision": True, "params": "7B"},
    "Qwen/Qwen2.5-VL-3B-Instruct": {"vision": True, "params": "3B"},
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {"vision": True, "params": "11B"},
    "microsoft/Phi-3.5-vision-instruct": {"vision": True, "params": "4.2B"},
    "meta-llama/Llama-3.1-8B-Instruct": {"vision": False, "params": "8B"},
    "Qwen/Qwen2.5-7B-Instruct": {"vision": False, "params": "7B"},
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    task: str = "task1"
    model: str = "qwen2.5-vl:7b"
    backend: str = "ollama"          # "ollama" or "vllm"
    prompt_strategy: str = "zero_shot"
    num_few_shot: int = 3
    output_dir: str = "results"
    data_dir: str = "data"
    temperature: float = 0.0
    max_tokens: int = 2048
    num_trials: int = 3
    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"


# ============================================================================
# Prompt Templates
# ============================================================================
TASK1_ZERO_SHOT = """You are a biomedical nano-device controller. Given a 2D sketch of a desired
intervention path within an anatomical structure, generate a sequence of 3D waypoints
that a nano-device should follow.

The sketch shows: {sketch_description}
Anatomical context: {anatomy_context}
Scale: {scale_info}

Output a JSON array of waypoints, each with keys "x", "y", "z" (in micrometers)
and "velocity" (in micrometers/second). Ensure the trajectory:
1. Stays within vessel/tissue boundaries
2. Avoids marked obstacles
3. Reaches the indicated target
4. Maintains smooth curvature suitable for magnetic actuation

Output ONLY valid JSON."""

TASK1_COT = """You are a biomedical nano-device controller. Analyze the following sketch
step-by-step to generate a precise 3D trajectory.

The sketch shows: {sketch_description}
Anatomical context: {anatomy_context}
Scale: {scale_info}

Follow these reasoning steps:
1. IDENTIFY: List all anatomical landmarks and structures visible in the sketch.
2. INTERPRET: Determine the intended action (drug delivery, tissue repair, etc.).
3. SPATIAL MAP: Convert 2D sketch positions to estimated 3D coordinates using the
   anatomical context.
4. PLAN PATH: Generate waypoints that navigate from start to target while respecting
   anatomical constraints.
5. VERIFY: Check that the trajectory is smooth, feasible, and safe.

After reasoning, output the final trajectory as a JSON array of waypoints with keys
"x", "y", "z" (micrometers) and "velocity" (micrometers/second)."""

TASK2_TEMPLATE = """You are an expert in biomedical spatial reasoning. Given the following
anatomical scene and nano-device position, answer the spatial relationship query.

Scene: {scene_description}
Device position: ({dev_x}, {dev_y}, {dev_z}) micrometers
Query: {query}

Provide your answer as JSON with keys:
- "answer": your spatial relationship answer
- "confidence": float between 0 and 1
- "reasoning": brief explanation"""

TASK3_TEMPLATE = """You are a real-time nano-device trajectory corrector. The device has
deviated from its planned path due to a disturbance.

Original trajectory target: ({target_x}, {target_y}, {target_z}) micrometers
Current device position: ({current_x}, {current_y}, {current_z}) micrometers
Disturbance: {disturbance_description}
Anatomical constraints: {constraints}

Generate a corrected trajectory (JSON array of waypoints with "x", "y", "z", "velocity")
from the current position to the target that:
1. Avoids the disturbance region
2. Maintains safe distance from anatomical boundaries
3. Minimizes total path length
4. Ensures smooth curvature for magnetic actuation

Output ONLY valid JSON."""


# ============================================================================
# Metrics (unchanged from original)
# ============================================================================
def compute_trajectory_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute RMSE between predicted and ground-truth waypoint sequences."""
    n_points = max(len(predicted), len(ground_truth))
    pred_resampled = np.array([
        np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(predicted)), predicted[:, i])
        for i in range(3)
    ]).T
    gt_resampled = np.array([
        np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(ground_truth)), ground_truth[:, i])
        for i in range(3)
    ]).T
    return float(np.sqrt(np.mean((pred_resampled - gt_resampled) ** 2)))


def compute_hausdorff_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute directed Hausdorff distance between two trajectories."""
    from scipy.spatial.distance import directed_hausdorff
    d1 = directed_hausdorff(predicted, ground_truth)[0]
    d2 = directed_hausdorff(ground_truth, predicted)[0]
    return float(max(d1, d2))


def compute_smoothness(waypoints: np.ndarray) -> float:
    """Compute mean curvature of trajectory (lower = smoother)."""
    if len(waypoints) < 3:
        return 0.0
    d1 = np.diff(waypoints, axis=0)
    d2 = np.diff(d1, axis=0)
    d1_mid = (d1[:-1] + d1[1:]) / 2
    d1_norm = np.linalg.norm(d1_mid, axis=1)
    d2_norm = np.linalg.norm(d2, axis=1)
    valid = d1_norm > 1e-10
    curvatures = np.zeros_like(d1_norm)
    curvatures[valid] = d2_norm[valid] / (d1_norm[valid] ** 2)
    return float(np.mean(curvatures))


def compute_feasibility(waypoints: np.ndarray, boundaries: dict) -> float:
    """Check what fraction of waypoints lie within anatomical boundaries."""
    center = np.array(boundaries.get("center", [0, 0]))
    radius = boundaries.get("radius", 100.0)
    xy_dist = np.sqrt((waypoints[:, 0] - center[0])**2 + (waypoints[:, 1] - center[1])**2)
    feasible = np.sum(xy_dist <= radius)
    return float(feasible / len(waypoints))


# ============================================================================
# Local Model Backends
# ============================================================================
class OllamaBackend:
    """Query models via Ollama REST API (runs locally on Jetson)."""

    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.model_name = model_name
        self.config = config
        self.base_url = config.ollama_url

    def query_text(self, prompt: str) -> tuple[str, float]:
        """Send text-only prompt to Ollama."""
        import requests
        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            response = result.get("response", "")
        except Exception as e:
            response = f'{{"error": "{str(e)}"}}'
        latency = time.time() - start
        return response, latency

    def query_vision(self, prompt: str, image_path: str) -> tuple[str, float]:
        """Send text + image prompt to Ollama (for VLMs)."""
        import requests
        start = time.time()
        try:
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode()

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            response = result.get("response", "")
        except Exception as e:
            response = f'{{"error": "{str(e)}"}}'
        latency = time.time() - start
        return response, latency

    def get_model_info(self) -> dict:
        """Get model metadata from Ollama."""
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=30,
            )
            return resp.json()
        except Exception:
            return {}


class VLLMBackend:
    """Query models via vLLM OpenAI-compatible API (runs locally on Jetson)."""

    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.model_name = model_name
        self.config = config
        self.base_url = config.vllm_url

    def query_text(self, prompt: str) -> tuple[str, float]:
        """Send text-only prompt to vLLM."""
        import requests
        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            response = result["choices"][0]["text"]
        except Exception as e:
            response = f'{{"error": "{str(e)}"}}'
        latency = time.time() - start
        return response, latency

    def query_vision(self, prompt: str, image_path: str) -> tuple[str, float]:
        """Send text + image via vLLM OpenAI-compatible chat API."""
        import requests
        start = time.time()
        try:
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode()

            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }}
                        ],
                    }],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            response = result["choices"][0]["message"]["content"]
        except Exception as e:
            response = f'{{"error": "{str(e)}"}}'
        latency = time.time() - start
        return response, latency


class HFTransformersBackend:
    """Direct Hugging Face Transformers inference (fallback, slower)."""

    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.model_name = model_name
        self.config = config
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                torch_dtype="auto",
            )
        return self._pipeline

    def query_text(self, prompt: str) -> tuple[str, float]:
        start = time.time()
        try:
            pipe = self._get_pipeline()
            result = pipe(prompt, max_new_tokens=self.config.max_tokens,
                         temperature=self.config.temperature or 0.01)
            response = result[0]["generated_text"][len(prompt):]
        except Exception as e:
            response = f'{{"error": "{str(e)}"}}'
        return response, time.time() - start

    def query_vision(self, prompt: str, image_path: str) -> tuple[str, float]:
        # For vision models, use Ollama or vLLM instead
        return self.query_text(f"[Image: {image_path}]\n{prompt}")


def get_backend(model_name: str, config: BenchmarkConfig):
    """Factory: select inference backend."""
    if config.backend == "ollama":
        return OllamaBackend(model_name, config)
    elif config.backend == "vllm":
        return VLLMBackend(model_name, config)
    elif config.backend == "transformers":
        return HFTransformersBackend(model_name, config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


def is_vision_model(model_name: str) -> bool:
    """Check if model supports vision input."""
    if model_name in OLLAMA_MODELS:
        return OLLAMA_MODELS[model_name]["vision"]
    if model_name in HF_MODELS:
        return HF_MODELS[model_name]["vision"]
    # Heuristic fallback
    vision_keywords = ["vl", "vision", "llava", "phi3.5-v"]
    return any(kw in model_name.lower() for kw in vision_keywords)


# ============================================================================
# Data Loading
# ============================================================================
def load_task_data(task: str, data_dir: str) -> list[dict]:
    """Load benchmark data for the specified task."""
    data_path = Path(data_dir) / f"{task}_data.jsonl"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print(f"Generate data first: python generate_sample_data.py --task {task}")
        return []
    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} instances for {task}")
    return data


# ============================================================================
# Evaluation Runners
# ============================================================================
def evaluate_task1(backend, data: list[dict], config: BenchmarkConfig) -> dict:
    """Evaluate Task 1: Sketch-to-3D Trajectory Translation."""
    results = []
    for instance in data:
        if config.prompt_strategy == "chain_of_thought":
            prompt = TASK1_COT.format(**instance)
        else:
            prompt = TASK1_ZERO_SHOT.format(**instance)

        has_image = instance.get("sketch_path") and os.path.exists(instance.get("sketch_path", ""))
        if has_image and is_vision_model(config.model):
            response, latency = backend.query_vision(prompt, instance["sketch_path"])
        else:
            response, latency = backend.query_text(prompt)

        try:
            # Try to extract JSON from response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                waypoints_raw = json.loads(response[json_start:json_end])
            else:
                waypoints_raw = json.loads(response)
            if isinstance(waypoints_raw, dict) and "waypoints" in waypoints_raw:
                waypoints_raw = waypoints_raw["waypoints"]
            predicted = np.array([[w["x"], w["y"], w["z"]] for w in waypoints_raw])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            print(f"  Failed to parse response for {instance['id']}")
            predicted = np.zeros((2, 3))

        gt = np.array(instance["ground_truth_waypoints"])
        boundaries = instance.get("boundaries", {"center": [0, 0], "radius": 100})

        result = {
            "id": instance["id"],
            "rmse": compute_trajectory_rmse(predicted, gt),
            "hausdorff": compute_hausdorff_distance(predicted, gt),
            "smoothness": compute_smoothness(predicted),
            "feasibility": compute_feasibility(predicted, boundaries),
            "latency_ms": latency * 1000,
            "num_waypoints": len(predicted),
        }
        results.append(result)
        print(f"  {instance['id']}: RMSE={result['rmse']:.2f}um, "
              f"Feas={result['feasibility']:.2%}, Lat={result['latency_ms']:.0f}ms")

    summary = {}
    if results:
        summary = {
            "mean_rmse": float(np.mean([r["rmse"] for r in results])),
            "mean_hausdorff": float(np.mean([r["hausdorff"] for r in results])),
            "mean_smoothness": float(np.mean([r["smoothness"] for r in results])),
            "mean_feasibility": float(np.mean([r["feasibility"] for r in results])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
            "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in results], 95)),
        }
    return {"instances": results, "summary": summary}


def evaluate_task2(backend, data: list[dict], config: BenchmarkConfig) -> dict:
    """Evaluate Task 2: Anatomical Spatial Relationship Reasoning."""
    results = []
    for instance in data:
        prompt = TASK2_TEMPLATE.format(
            scene_description=instance["scene_description"],
            dev_x=instance["device_position"][0],
            dev_y=instance["device_position"][1],
            dev_z=instance["device_position"][2],
            query=instance["query"],
        )
        response, latency = backend.query_text(prompt)
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
            else:
                parsed = json.loads(response)
            predicted_answer = parsed.get("answer", "")
            confidence = parsed.get("confidence", 0.5)
        except (json.JSONDecodeError, KeyError):
            predicted_answer = response.strip()
            confidence = 0.5

        correct = predicted_answer.lower().strip() == instance["ground_truth_answer"].lower().strip()
        result = {
            "id": instance["id"],
            "correct": correct,
            "confidence": float(confidence),
            "latency_ms": latency * 1000,
        }
        results.append(result)

    summary = {}
    if results:
        summary = {
            "accuracy": float(np.mean([r["correct"] for r in results])),
            "mean_confidence": float(np.mean([r["confidence"] for r in results])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
        }
    return {"instances": results, "summary": summary}


def evaluate_task3(backend, data: list[dict], config: BenchmarkConfig) -> dict:
    """Evaluate Task 3: Closed-Loop Trajectory Correction."""
    results = []
    for instance in data:
        prompt = TASK3_TEMPLATE.format(
            target_x=instance["target"][0],
            target_y=instance["target"][1],
            target_z=instance["target"][2],
            current_x=instance["current_position"][0],
            current_y=instance["current_position"][1],
            current_z=instance["current_position"][2],
            disturbance_description=instance["disturbance"],
            constraints=instance["constraints"],
        )
        response, latency = backend.query_text(prompt)
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                waypoints_raw = json.loads(response[json_start:json_end])
            else:
                waypoints_raw = json.loads(response)
            if isinstance(waypoints_raw, dict) and "waypoints" in waypoints_raw:
                waypoints_raw = waypoints_raw["waypoints"]
            predicted = np.array([[w["x"], w["y"], w["z"]] for w in waypoints_raw])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            predicted = np.zeros((2, 3))

        gt = np.array(instance["ground_truth_correction"])
        result = {
            "id": instance["id"],
            "correction_rmse": compute_trajectory_rmse(predicted, gt),
            "latency_ms": latency * 1000,
            "smoothness": compute_smoothness(predicted),
        }
        results.append(result)

    summary = {}
    if results:
        summary = {
            "mean_correction_rmse": float(np.mean([r["correction_rmse"] for r in results])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
            "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in results], 95)),
            "mean_smoothness": float(np.mean([r["smoothness"] for r in results])),
        }
    return {"instances": results, "summary": summary}


# ============================================================================
# Jetson Hardware Profiling
# ============================================================================
def profile_jetson() -> dict:
    """Collect Jetson hardware stats (tegrastats-based)."""
    import subprocess
    info = {"platform": "Jetson AGX Orin 64GB"}
    try:
        result = subprocess.run(
            ["cat", "/etc/nv_tegra_release"], capture_output=True, text=True, timeout=5
        )
        info["tegra_release"] = result.stdout.strip()
    except Exception:
        info["tegra_release"] = "N/A"
    try:
        result = subprocess.run(
            ["cat", "/proc/meminfo"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "MemTotal" in line:
                info["ram_total_kb"] = int(line.split()[1])
            if "MemAvailable" in line:
                info["ram_available_kb"] = int(line.split()[1])
    except Exception:
        pass
    return info


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="NanoSpatialBench — Jetson Edge Evaluation")
    parser.add_argument("--task", type=str, default="task1",
                        choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--model", type=str, default="qwen2.5-vl:7b",
                        help="Model name (Ollama tag or HF ID), or 'all'")
    parser.add_argument("--backend", type=str, default="ollama",
                        choices=["ollama", "vllm", "transformers"])
    parser.add_argument("--prompt_strategy", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "chain_of_thought"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    config = BenchmarkConfig(
        task=args.task, model=args.model, backend=args.backend,
        prompt_strategy=args.prompt_strategy, output_dir=args.output_dir,
        data_dir=args.data_dir, ollama_url=args.ollama_url, vllm_url=args.vllm_url,
    )
    os.makedirs(config.output_dir, exist_ok=True)

    # Determine model list
    if args.model == "all":
        model_list = list(OLLAMA_MODELS.keys()) if args.backend == "ollama" else list(HF_MODELS.keys())
    else:
        model_list = [args.model]

    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    evaluators = {"task1": evaluate_task1, "task2": evaluate_task2, "task3": evaluate_task3}

    # Profile hardware
    hw_info = profile_jetson()
    print(f"Hardware: {hw_info.get('platform', 'unknown')}")

    for model_name in model_list:
        config.model = model_name
        backend = get_backend(model_name, config)

        all_results = {}
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Task: {task} | Strategy: {config.prompt_strategy}")
            print(f"Backend: {config.backend} | Device: Jetson AGX Orin 64GB")
            print(f"{'='*60}")

            data = load_task_data(task, config.data_dir)
            if not data:
                continue

            results = evaluators[task](backend, data, config)
            all_results[task] = results

            print(f"\nSummary for {task}:")
            for k, v in results["summary"].items():
                print(f"  {k}: {v:.4f}")

        # Save results
        safe_name = model_name.replace("/", "_").replace(":", "_")
        output_path = Path(config.output_dir) / f"{safe_name}_{config.prompt_strategy}_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "config": asdict(config),
                "hardware": hw_info,
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
