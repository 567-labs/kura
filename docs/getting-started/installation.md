# Installation Guide

This guide will walk you through the installation process for Kura.

## Requirements

Kura has the following requirements:

- Python 3.9+ (Python 3.9 is specifically recommended due to UMAP dependency)
- uv package manager
- OpenAI API key for model access

## Installation

```bash
# Install using uv
uv pip install kura
```

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

## Installing Optional Dependencies

Kura supports additional features with optional dependencies:

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
