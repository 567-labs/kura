# Installation Guide

This guide will walk you through the installation process for Kura.

## Requirements

Kura has the following requirements:

- Python 3.9+ (Python 3.9 is specifically recommended due to UMAP dependency)
- uv package manager
- OpenAI API key for model access

## Installation

### Basic Installation

```bash
# Install using uv (includes JSONL checkpoint manager only)
uv pip install kura
```

### Optional Dependencies

For advanced checkpoint managers that provide better compression and performance:

```bash
# Install with checkpoint support (HuggingFace Datasets and Parquet managers)
uv pip install "kura[checkpoints]"

# Install with all optional dependencies
uv pip install "kura[all]"
```

**Available Optional Dependencies:**

- **`checkpoints`**: Enables HuggingFace Datasets and Parquet checkpoint managers
  - Includes: `datasets>=3.6.0`, `pyarrow>=10.0.0`
  - Recommended for production use and large datasets (>5k conversations)
- **`all`**: All optional dependencies combined

### Development Installation

If you want to contribute to Kura or modify the source code, install it in development mode:

```bash
# Clone the repository
git clone https://github.com/567-labs/kura.git
cd kura

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
uv pip install -e . --group dev
```

## Setting up API Keys

Kura uses OpenAI models for processing. You'll need to set up an API key:

1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:

```bash
# On Linux/macOS
export OPENAI_API_KEY=your_api_key_here

# On Windows
set OPENAI_API_KEY=your_api_key_here
```

## Installing Additional Development Dependencies

For development work, you can install all development and documentation dependencies:

```bash
uv sync --all-extras --group dev --group docs
```

## Verifying Your Installation

To verify that Kura is installed correctly, run:

```bash
python -c "from kura import summarise_conversations; print('Kura installed successfully')"
```

You should see a confirmation message with no errors.

## Next Steps

Now that you have Kura installed, proceed to the [Quickstart guide](quickstart.md) to begin analyzing your first dataset.
