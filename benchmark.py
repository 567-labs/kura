#!/usr/bin/env python3
"""Import performance benchmark for Kura."""

import time
import sys


def main():
    print('=== Kura Import Benchmark ===')
    print(f'Python version: {sys.version}')
    print()
    
    # Benchmark individual module imports
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
        start_time = time.perf_counter()
        try:
            __import__(module)
            import_time = time.perf_counter() - start_time
            total_time += import_time
            print(f'{module}: {import_time:.4f}s')
        except Exception as e:
            print(f'{module}: failed - {e}')
    
    print()
    print(f'Total import time: {total_time:.4f}s')
    print('========================')


if __name__ == '__main__':
    main()
