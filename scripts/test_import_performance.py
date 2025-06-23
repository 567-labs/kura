#!/usr/bin/env python3
"""
Test script to measure import performance of kura package.
This script measures the time taken to import kura and its submodules.
"""

import time
import sys
import json
import subprocess
from pathlib import Path


def measure_import_time(module_name: str) -> float:
    """Measure the time taken to import a module in a fresh Python process."""
    import_code = f"""
import time
start_time = time.perf_counter()
import {module_name}
end_time = time.perf_counter()
print(end_time - start_time)
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", import_code],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            print(f"Error importing {module_name}: {result.stderr}")
            return -1
    except subprocess.TimeoutExpired:
        print(f"Import of {module_name} timed out after 60 seconds")
        return -1
    except Exception as e:
        print(f"Unexpected error importing {module_name}: {e}")
        return -1


def measure_individual_modules():
    """Measure import time for individual kura modules."""
    modules_to_test = [
        "kura",
        "kura.summarisation", 
        "kura.embedding",
        "kura.cluster",
        "kura.visualization",
        "kura.checkpoint",
        "kura.k_means",
        "kura.hdbscan",
    ]
    
    results = {}
    
    for module in modules_to_test:
        print(f"Testing import of {module}...")
        import_time = measure_import_time(module)
        results[module] = import_time
        
        if import_time > 0:
            print(f"  {module}: {import_time:.3f}s")
        else:
            print(f"  {module}: FAILED")
    
    return results


def measure_step_by_step_import():
    """Measure cumulative import time by importing modules step by step."""
    import_sequence = [
        "import sys",
        "import asyncio", 
        "import logging",
        "import numpy as np",
        "import instructor",
        "from openai import AsyncOpenAI",
        "import kura.types",
        "import kura.base_classes", 
        "import kura.utils",
        "import kura.checkpoint",
        "import kura.embedding",
        "import kura.summarisation",
        "import kura.cluster",
        "import kura"
    ]
    
    results = {}
    cumulative_code = ""
    
    for step in import_sequence:
        cumulative_code += step + "\n"
        
        test_code = f"""
import time
start_time = time.perf_counter()
{cumulative_code}
end_time = time.perf_counter()
print(end_time - start_time)
"""
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                import_time = float(result.stdout.strip())
                results[step] = import_time
                print(f"After '{step}': {import_time:.3f}s")
            else:
                print(f"Error at step '{step}': {result.stderr}")
                results[step] = -1
                
        except Exception as e:
            print(f"Error testing step '{step}': {e}")
            results[step] = -1
    
    return results


def main():
    print("=" * 60)
    print("KURA IMPORT PERFORMANCE TEST")
    print("=" * 60)
    
    # Test 1: Overall kura import time
    print("\n1. Testing overall kura import time...")
    kura_import_time = measure_import_time("kura")
    
    if kura_import_time > 0:
        print(f"   Total kura import time: {kura_import_time:.3f}s")
        
        # Flag slow imports
        if kura_import_time > 5.0:
            print("   ⚠️  WARNING: Import time is very slow (>5s)")
        elif kura_import_time > 2.0:
            print("   ⚠️  NOTICE: Import time is slow (>2s)")
        else:
            print("   ✅ Import time is acceptable")
    else:
        print("   ❌ Failed to import kura")
    
    # Test 2: Individual module imports
    print("\n2. Testing individual module imports...")
    individual_results = measure_individual_modules()
    
    # Test 3: Step-by-step import analysis
    print("\n3. Step-by-step import analysis...")
    step_results = measure_step_by_step_import()
    
    # Compile results
    all_results = {
        "overall_import_time": kura_import_time,
        "individual_modules": individual_results,
        "step_by_step": step_results,
        "python_version": sys.version,
        "test_timestamp": time.time()
    }
    
    # Save results to file
    results_file = Path("import_performance_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall kura import: {kura_import_time:.3f}s")
    
    # Find slowest individual modules
    valid_modules = {k: v for k, v in individual_results.items() if v > 0}
    if valid_modules:
        slowest_module = max(valid_modules.items(), key=lambda x: x[1])
        print(f"Slowest module: {slowest_module[0]} ({slowest_module[1]:.3f}s)")
    
    # Exit with error code if import is too slow
    if kura_import_time > 10.0:
        print("❌ CRITICAL: Import time exceeds 10 seconds")
        sys.exit(1)
    elif kura_import_time > 5.0:
        print("⚠️  WARNING: Import time exceeds 5 seconds") 
        sys.exit(1)
    else:
        print("✅ Import performance test passed")


if __name__ == "__main__":
    main()
