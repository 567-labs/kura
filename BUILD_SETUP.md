# Build Setup for Including Vite Assets

This document explains how the Kura project is configured to include Vite-built UI assets in the Python package when running `uv build`.

## Configuration Changes Made

### 1. Modified `pyproject.toml`

Changed the build system from hatchling to setuptools to better support including additional files:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["kura*"]

[tool.setuptools.package-data]
kura = ["static/**/*"]
```

### 2. Vite Configuration

The Vite project in `ui/` is already configured to build to the correct location:

```typescript
// ui/vite.config.ts
export default defineConfig({
  // ...
  build: {
    outDir: path.resolve(__dirname, "../kura/static/dist"),
    emptyOutDir: true,
  },
});
```

### 3. Build Script

Created `scripts/build_package.sh` to automate the complete build process:

```bash
#!/bin/bash
set -e

echo "Building UI..."
cd ui
npm install
npm run build
cd ..

echo "Building Python package..."
rm -rf dist build *.egg-info
uv build

echo "Build complete! Static files included:"
unzip -l dist/kura-*.whl | grep static || echo "No static files found in wheel"
```

## How It Works

1. **Vite Build**: The UI project builds to `kura/static/dist/`
2. **Package Data**: setuptools is configured to include all files under `kura/static/`
3. **Automatic Inclusion**: When `uv build` runs, it automatically includes the static files in both the source distribution and wheel

## Build Workflow

### Manual Steps
```bash
# Build UI
cd ui
npm install
npm run build
cd ..

# Build Python package
uv build
```

### Automated Script
```bash
# Use the provided script
./scripts/build_package.sh
```

## Verification

To verify that static files are included in the built package:

```bash
# Check wheel contents
unzip -l dist/kura-*.whl | grep static

# Extract and verify structure
unzip -q dist/kura-*.whl -d extracted
find extracted/kura/static -type f
```

## Files Included

The following Vite-built assets are included in the package:
- `kura/static/dist/index.html`
- `kura/static/dist/vite.svg`
- `kura/static/dist/assets/*.css`
- `kura/static/dist/assets/*.js`

## Important Notes

1. The UI must be built before running `uv build` for the latest changes to be included
2. The setuptools configuration automatically includes all files under `kura/static/`
3. No MANIFEST.in file is needed with this setup
4. The build script ensures the UI is always up-to-date when building the package