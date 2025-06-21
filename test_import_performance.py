#!/usr/bin/env python3
"""
Simple test to check if our import optimizations work.
"""

import time
import sys

def test_import():
    """Test the import time of kura module."""
    print("Testing import time of kura module...")
    
    start_time = time.time()
    try:
        import kura
        import_time = time.time() - start_time
        print(f"✅ Successfully imported kura in {import_time:.3f} seconds")
        
        # Test that classes are available
        print(f"✅ KmeansClusteringMethod available: {hasattr(kura, 'KmeansClusteringMethod')}")
        print(f"✅ HDBSCANClusteringMethod available: {hasattr(kura, 'HDBSCANClusteringMethod')}")
        print(f"✅ SummaryModel available: {hasattr(kura, 'SummaryModel')}")
        
        # Test instantiation (this would trigger the actual import)
        print("\nTesting instantiation (this will trigger actual imports):")
        try:
            kmeans = kura.KmeansClusteringMethod()
            print("✅ KmeansClusteringMethod instantiated successfully")
        except ImportError as e:
            print(f"⚠️  KmeansClusteringMethod requires optional dependency: {e}")
        
        try:
            hdbscan = kura.HDBSCANClusteringMethod()
            print("✅ HDBSCANClusteringMethod instantiated successfully") 
        except ImportError as e:
            print(f"⚠️  HDBSCANClusteringMethod requires optional dependency: {e}")
            
    except Exception as e:
        import_time = time.time() - start_time
        print(f"❌ Failed to import kura after {import_time:.3f} seconds: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_import()
    if success:
        print("\n🎉 Import optimization test completed successfully!")
    else:
        print("\n❌ Import optimization test failed!")
        sys.exit(1)