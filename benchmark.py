#!/usr/bin/env python3
"""Import performance benchmark for Kura."""

import subprocess
import sys


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
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30
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


def main():
    print('=== Kura Import Benchmark (Isolated) ===')
    print(f'Python version: {sys.version}')
    print()
    
    # Benchmark individual module imports in isolation
    modules_to_test = [
        'kura.types',
        'kura.k_means',
        'kura.hdbscan', 
        'kura.embedding',
        'kura.summarisation',
        'kura'
    ]
    
    total_time = 0
    
    for module in modules_to_test:
        import_time, error = test_import_time(module)
        if import_time is not None:
            total_time += import_time
            print(f'{module}: {import_time:.4f}s')
        else:
            print(f'{module}: failed - {error}')
    
    print()
    print(f'Total import time: {total_time:.4f}s')
    print('========================')


if __name__ == '__main__':
    main()
