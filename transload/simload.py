#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray


@dataclass
class EventSpec:
    event_id: int
    arrival_time_s: float

    total_work_s: float
    cpu_fraction: float
    gpu_fraction: float

    cpu_work_s: float
    gpu_work_s: float

    cpu_matrix_size: int
    gpu_matrix_size: int

    gpu_memory_mb: int
    h2d_mb: int
    d2h_mb: int

    num_cpus: float
    num_gpus: float


def sample_event(
    event_id: int,
    arrival_time_s: float,
    params: Dict[str, Any],
) -> EventSpec:
    """
    Generate one synthetic Geant4-like event.

    The CPU/GPU split is randomized using a Beta distribution.
    """

    rng = random.Random(params.get("seed", 12345) + event_id)

    mean_total_work_s = float(params.get("mean_total_work_s", 3.0))
    total_work_jitter = float(params.get("total_work_jitter", 0.5))

    # Lognormal gives positive, event-to-event variable work.
    sigma = total_work_jitter
    mu = math.log(mean_total_work_s) - 0.5 * sigma * sigma
    total_work_s = rng.lognormvariate(mu, sigma)

    beta_alpha = float(params.get("cpu_fraction_beta_alpha", 4.0))
    beta_beta = float(params.get("cpu_fraction_beta_beta", 4.0))

    cpu_fraction = rng.betavariate(beta_alpha, beta_beta)
    gpu_fraction = 1.0 - cpu_fraction

    cpu_work_s = total_work_s * cpu_fraction
    gpu_work_s = total_work_s * gpu_fraction

    # Matrix sizes loosely correlated with work.
    cpu_matrix_base = int(params.get("cpu_matrix_base", 512))
    gpu_matrix_base = int(params.get("gpu_matrix_base", 2048))

    cpu_matrix_size = max(
        128,
        int(cpu_matrix_base * math.sqrt(max(cpu_work_s, 0.05))),
    )
    gpu_matrix_size = max(
        256,
        int(gpu_matrix_base * math.sqrt(max(gpu_work_s, 0.05))),
    )

    # Optional memory and transfer sizes correlated with GPU fraction.
    gpu_memory_base_mb = int(params.get("gpu_memory_base_mb", 256))
    gpu_memory_jitter_mb = int(params.get("gpu_memory_jitter_mb", 256))

    gpu_memory_mb = int(
        gpu_memory_base_mb * gpu_fraction
        + rng.uniform(0, gpu_memory_jitter_mb) * gpu_fraction
    )

    h2d_mb = int(params.get("h2d_base_mb", 64) * gpu_fraction)
    d2h_mb = int(params.get("d2h_base_mb", 16) * gpu_fraction)

    return EventSpec(
        event_id=event_id,
        arrival_time_s=arrival_time_s,
        total_work_s=total_work_s,
        cpu_fraction=cpu_fraction,
        gpu_fraction=gpu_fraction,
        cpu_work_s=cpu_work_s,
        gpu_work_s=gpu_work_s,
        cpu_matrix_size=cpu_matrix_size,
        gpu_matrix_size=gpu_matrix_size,
        gpu_memory_mb=gpu_memory_mb,
        h2d_mb=h2d_mb,
        d2h_mb=d2h_mb,
        num_cpus=float(params.get("num_cpus_per_event", 1.0)),
        num_gpus=float(params.get("num_gpus_per_event", 1.0)),
    )


@ray.remote
def cpu_stage(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthetic CPU stage.

    This represents Geant4-like CPU work:
      - tracking
      - stepping
      - geometry navigation
      - physics process bookkeeping
      - hit creation
    """

    start = time.perf_counter()

    target_s = float(event["cpu_work_s"])
    matrix_size = int(event["cpu_matrix_size"])

    if target_s <= 0.0:
        end = time.perf_counter()
        return {
            "event_id": event["event_id"],
            "stage": "cpu",
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "iterations": 0,
        }

    a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    iterations = 0
    checksum = 0.0

    while time.perf_counter() - start < target_s:
        c = a @ b
        checksum += float(c[0, 0])
        iterations += 1

    end = time.perf_counter()

    return {
        "event_id": event["event_id"],
        "stage": "cpu",
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "iterations": iterations,
        "checksum": checksum,
    }


@ray.remote
def gpu_stage(event: Dict[str, Any], cpu_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Synthetic GPU stage.

    This represents GPU-side work such as:
      - optical photon propagation
      - large vectorized scoring
      - batched hit processing
      - detector response kernels

    The cpu_result argument creates a dependency:
    GPU work for this event cannot start until the CPU stage has completed.
    """

    import torch

    start = time.perf_counter()

    target_s = float(event["gpu_work_s"])
    matrix_size = int(event["gpu_matrix_size"])
    gpu_memory_mb = int(event["gpu_memory_mb"])
    h2d_mb = int(event["h2d_mb"])
    d2h_mb = int(event["d2h_mb"])

    if target_s <= 0.0:
        end = time.perf_counter()
        return {
            "event_id": event["event_id"],
            "stage": "gpu",
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "iterations": 0,
        }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    device = torch.device("cuda")

    # Simulate host-to-device transfer.
    if h2d_mb > 0:
        n = h2d_mb * 1024 * 1024 // 4
        host_data = torch.rand((n,), dtype=torch.float32, pin_memory=True)
        device_data = host_data.to(device, non_blocking=False)
        torch.cuda.synchronize()
    else:
        device_data = None

    # Simulate event-level GPU memory footprint.
    memory_block = None
    if gpu_memory_mb > 0:
        n = gpu_memory_mb * 1024 * 1024 // 4
        memory_block = torch.empty((n,), device=device, dtype=torch.float32)
        memory_block.fill_(1.0)
        torch.cuda.synchronize()

    a = torch.rand((matrix_size, matrix_size), device=device, dtype=torch.float32)
    b = torch.rand((matrix_size, matrix_size), device=device, dtype=torch.float32)

    iterations = 0
    checksum = 0.0

    while time.perf_counter() - start < target_s:
        c = a @ b
        torch.cuda.synchronize()
        checksum += float(c[0, 0].item())
        iterations += 1

    # Simulate device-to-host transfer.
    if d2h_mb > 0:
        n = d2h_mb * 1024 * 1024 // 4
        out = torch.empty((n,), device=device, dtype=torch.float32)
        host_out = out.cpu()
        torch.cuda.synchronize()
        checksum += float(host_out[0].item())

    if memory_block is not None:
        checksum += float(memory_block[0].item())

    if device_data is not None:
        checksum += float(device_data[0].item())

    end = time.perf_counter()

    return {
        "event_id": event["event_id"],
        "stage": "gpu",
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "iterations": iterations,
        "checksum": checksum,
    }


@ray.remote
def event_reduce_stage(
    event: Dict[str, Any],
    cpu_result: Dict[str, Any],
    gpu_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Lightweight event-level reduction.

    This represents merging CPU/GPU results into an event record.
    """

    start = time.perf_counter()

    # Small bookkeeping delay.
    time.sleep(float(event.get("reduce_work_s", 0.01)))

    end = time.perf_counter()

    return {
        "event_id": event["event_id"],
        "stage": "reduce",
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "cpu_runtime_s": cpu_result["runtime_s"],
        "gpu_runtime_s": gpu_result["runtime_s"],
    }


def generate_events(params: Dict[str, Any]) -> List[EventSpec]:
    n_events = int(params.get("n_events", 20))
    arrival_mode = params.get("arrival_mode", "batch")

    events = []

    if arrival_mode == "batch":
        for event_id in range(n_events):
            events.append(sample_event(event_id, 0.0, params))

    elif arrival_mode == "poisson":
        mean_interarrival_s = float(params.get("mean_interarrival_s", 0.5))
        rng = random.Random(params.get("seed", 12345))

        t = 0.0
        for event_id in range(n_events):
            t += rng.expovariate(1.0 / mean_interarrival_s)
            events.append(sample_event(event_id, t, params))

    elif arrival_mode == "fixed":
        interarrival_s = float(params.get("interarrival_s", 0.5))

        for event_id in range(n_events):
            events.append(sample_event(event_id, event_id * interarrival_s, params))

    else:
        raise ValueError(f"Unknown arrival_mode: {arrival_mode}")

    return events


def run_simulation(params: Dict[str, Any]) -> pd.DataFrame:
    events = generate_events(params)

    t0 = time.perf_counter()

    result_refs = []
    event_records = []

    for spec in events:
        target_submit_time = t0 + spec.arrival_time_s
        now = time.perf_counter()

        if target_submit_time > now:
            time.sleep(target_submit_time - now)

        event = asdict(spec)
        event["reduce_work_s"] = float(params.get("reduce_work_s", 0.01))

        submitted_at = time.perf_counter()

        cpu_ref = cpu_stage.options(
            num_cpus=spec.num_cpus,
            num_gpus=0,
        ).remote(event)

        # GPU stage depends on CPU stage by default.
        # This resembles Geant4 CPU event preparation followed by GPU optical work.
        if bool(params.get("gpu_depends_on_cpu", True)):
            gpu_ref = gpu_stage.options(
                num_cpus=float(params.get("num_cpus_per_gpu_task", 0.25)),
                num_gpus=spec.num_gpus,
            ).remote(event, cpu_ref)
        else:
            gpu_ref = gpu_stage.options(
                num_cpus=float(params.get("num_cpus_per_gpu_task", 0.25)),
                num_gpus=spec.num_gpus,
            ).remote(event, None)

        reduce_ref = event_reduce_stage.options(
            num_cpus=float(params.get("num_cpus_per_reduce_task", 0.1)),
            num_gpus=0,
        ).remote(event, cpu_ref, gpu_ref)

        result_refs.append(reduce_ref)

        event_records.append(
            {
                "event_id": spec.event_id,
                "submitted_at": submitted_at,
                "submit_offset_s": submitted_at - t0,
                **asdict(spec),
            }
        )

        print(
            f"submitted event={spec.event_id:04d} "
            f"at={submitted_at - t0:8.3f}s "
            f"cpu_frac={spec.cpu_fraction:5.2f} "
            f"gpu_frac={spec.gpu_fraction:5.2f} "
            f"cpu_work={spec.cpu_work_s:6.3f}s "
            f"gpu_work={spec.gpu_work_s:6.3f}s "
            f"gpu_mem={spec.gpu_memory_mb:5d} MB",
            flush=True,
        )

    reduce_results = ray.get(result_refs)

    events_df = pd.DataFrame(event_records)
    results_df = pd.DataFrame(reduce_results)

    df = events_df.merge(results_df, on="event_id", how="left")

    df["end_offset_s"] = df["end_time"] - t0
    df["event_latency_s"] = df["end_time"] - df["submitted_at"]
    df["queue_plus_runtime_s"] = df["event_latency_s"]

    return df.sort_values("event_id")


def default_params() -> Dict[str, Any]:
    return {
        "seed": 12345,

        "n_events": 20,
        "arrival_mode": "fixed",
        "interarrival_s": 0.25,

        "mean_total_work_s": 3.0,
        "total_work_jitter": 0.6,

        # Beta(alpha, beta) for CPU fraction.
        # alpha=beta gives balanced events.
        # alpha > beta gives CPU-heavy events.
        # alpha < beta gives GPU-heavy events.
        "cpu_fraction_beta_alpha": 4.0,
        "cpu_fraction_beta_beta": 4.0,

        "cpu_matrix_base": 512,
        "gpu_matrix_base": 2048,

        "gpu_memory_base_mb": 256,
        "gpu_memory_jitter_mb": 512,
        "h2d_base_mb": 64,
        "d2h_base_mb": 16,

        "num_cpus_per_event": 1.0,
        "num_gpus_per_event": 1.0,
        "num_cpus_per_gpu_task": 0.25,
        "num_cpus_per_reduce_task": 0.1,

        "gpu_depends_on_cpu": True,
        "reduce_work_s": 0.01,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=None, help="JSON parameter file")
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=float, default=None)
    parser.add_argument("--out", default="geant4_like_ray_results.csv")
    args = parser.parse_args()

    params = default_params()

    if args.params is not None:
        with open(args.params, "r", encoding="utf-8") as f:
            user_params = json.load(f)
        params.update(user_params)

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    print("Ray resources:")
    print(ray.cluster_resources())
    print()

    df = run_simulation(params)

    cols = [
        "event_id",
        "submit_offset_s",
        "end_offset_s",
        "event_latency_s",
        "total_work_s",
        "cpu_fraction",
        "gpu_fraction",
        "cpu_work_s",
        "gpu_work_s",
        "gpu_memory_mb",
        "h2d_mb",
        "d2h_mb",
    ]

    print()
    print(df[cols].to_string(index=False))

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    ray.shutdown()


if __name__ == "__main__":
    main()
