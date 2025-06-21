#!/usr/bin/env python3
"""
Test script to measure import times for different Kura modules.
This helps identify which imports are causing the slow startup.
"""

import time
import sys
from contextlib import contextmanager

@contextmanager
def time_import(name):
    """Context manager to time imports."""
    start = time.time()
    print(f"Importing {name}...", end="", flush=True)
    yield
    duration = time.time() - start
    print(f" took {duration:.3f}s")

def main():
    print("Testing individual Kura module import times:")
    print("=" * 50)
    
    # Test individual module imports
    with time_import("hdbscan"):
        import hdbscan
    
    with time_import("openai"):
        from openai import AsyncOpenAI
    
    with time_import("instructor"):
        import instructor
    
    with time_import("rich"):
        try:
            from rich.console import Console
        except ImportError:
            print(" (not available)")
    
    with time_import("diskcache"):
        import diskcache
    
    with time_import("tenacity"):
        from tenacity import retry, wait_fixed, stop_after_attempt
        
    with time_import("tqdm"):
        from tqdm.asyncio import tqdm_asyncio
    
    print("\n" + "=" * 50)
    print("Testing full Kura import:")
    
    with time_import("kura (full)"):
        import kura
    
    print("\nTesting specific kura submodules:")
    
    with time_import("kura.hdbscan"):
        from kura import HDBSCANClusteringMethod
        
    with time_import("kura.embedding"):
        from kura.embedding import OpenAIEmbeddingModel
        
    with time_import("kura.summarisation"):
        from kura.summarisation import SummaryModel

if __name__ == "__main__":
    main()