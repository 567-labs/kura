"""
Hatchling build hook to ensure frontend is built before packaging.
"""

from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to build frontend before packaging."""

    def initialize(self, version, build_data):
        """Initialize the build hook."""
        static_dir = Path(self.root) / "kura" / "static" / "dist"

        # Verify static directory exists
        if not static_dir.exists() or not any(static_dir.iterdir()):
            raise FileNotFoundError(
                f"Static directory not found: {static_dir}. "
                "Please build the frontend first by running 'bun run build' in the ui/ directory."
            )
        
        file_count = len(list(static_dir.rglob('*')))
        print(f"Static directory found: {static_dir} ({file_count} files)")
