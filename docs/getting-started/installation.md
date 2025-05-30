# Installation Guide

This guide will walk you through the installation process for Kura.

## Requirements

Kura has the following requirements:

* Python 3.9+ (Python 3.9 is specifically recommended due to UMAP dependency)
* pip or uv package manager
* Google API key for Gemini model access (default summarization model)

## Installation Methods

=== "Using uv (Recommended)"

    ```bash
    # Install using uv for better performance
    uv pip install kura
    ```

=== "Using pip"

    ```bash
    # Install using standard pip
    pip install kura
    ```

### Development Installation

If you want to contribute to Kura or modify the source code, install it in development mode:

=== "Using uv (Recommended)"

    ```bash
    # Clone the repository
    git clone https://github.com/567-labs/kura.git
    cd kura
    
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install in development mode with dev dependencies
    uv pip install -e ".[dev]"
    ```

=== "Using pip"

    ```bash
    # Clone the repository
    git clone https://github.com/567-labs/kura.git
    cd kura
    
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install in development mode with dev dependencies
    pip install -e ".[dev]"
    ```

## Setting up API Keys

Kura uses the Gemini model from Google for summarization by default. You'll need to set up an API key:

1. Get a Google API key from [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
2. Set the environment variable:

```bash
# On Linux/macOS
export GOOGLE_API_KEY=your_api_key_here

# On Windows
set GOOGLE_API_KEY=your_api_key_here
```

## Installing Optional Dependencies

Kura supports additional features with optional dependencies:

=== "Using uv (Recommended)"

    ```bash
    # For documentation development
    uv pip install -e ".[docs]"
    
    # For running tests
    uv pip install -e ".[dev]"
    ```

=== "Using pip"

    ```bash
    # For documentation development
    pip install -e ".[docs]"
    
    # For running tests
    pip install -e ".[dev]"
    ```

## Verifying Your Installation

To verify that Kura is installed correctly, run:

```bash
python -c "from kura import Kura; print('Kura installed successfully')"
```

You should see a confirmation message with no errors.

## Next Steps

Now that you have Kura installed, continue with the [Procedural API quickstart](tutorial-procedural-api.md) to analyze your first dataset.
