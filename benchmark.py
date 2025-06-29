#!/usr/bin/env python3
"""Import performance benchmark for Kura."""

import subprocess
import sys
import statistics
import concurrent.futures
from typing import List, Tuple, Optional


def test_import_time(module_name):
    """Test import time for a module in isolation using subprocess."""
    code = f"""
import time
start = time.perf_counter()
try:
    import {module_name}
    print(time.perf_counter() - start)
except Exception as e:
    print(f"ERROR: {{e}}")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if output.startswith("ERROR"):
                return None, output
            return float(output), None
        else:
            return None, f"Failed with code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return None, "Timeout (>30s)"
    except Exception as e:
        return None, f"Subprocess error: {e}"


def test_import_multiple_times(
    module_name: str, num_runs: int = 5
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Test import time for a module multiple times in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_runs) as executor:
        # Submit all tasks
        futures = [
            executor.submit(test_import_time, module_name) for _ in range(num_runs)
        ]

        times = []
        errors = []

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            import_time, error = future.result()
            if import_time is not None:
                times.append(import_time)
            else:
                errors.append(error)

        if not times:
            return None, f"All runs failed: {errors}"

        if errors:
            print(f"  Warning: {len(errors)} out of {num_runs} runs failed")

        return times, None


def main():
    print("=== Kura Import Benchmark (5 runs per module) ===")
    print(f"Python version: {sys.version}")
    print()

    # Benchmark individual module imports in isolation
    modules_to_test = [
        "kura.types",
        "kura.k_means",
        "kura.hdbscan",
        "kura.embedding",
        "kura.summarisation",
        "kura",
    ]

    total_mean_time = 0
    results = {}

    for module in modules_to_test:
        print(f"Testing {module}...")
        times, error = test_import_multiple_times(module, num_runs=5)

        if times is not None:
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)

            total_mean_time += mean_time
            results[module] = {
                "mean": mean_time,
                "std": std_time,
                "min": min_time,
                "max": max_time,
                "times": times,
            }

            print(
                f"{module}: {mean_time:.4f}s ± {std_time:.4f}s (min: {min_time:.4f}s, max: {max_time:.4f}s)"
            )
            print(f"  Individual times: {[f'{t:.4f}s' for t in times]}")
        else:
            print(f"{module}: failed - {error}")

    print()
    print("=== Summary ===")
    for module, stats in results.items():
        print(f"{module}: {stats['mean']:.4f}s ± {stats['std']:.4f}s")

    print()
    print(f"Total mean import time: {total_mean_time:.4f}s")
    print("========================")


if __name__ == "__main__":
    main()
