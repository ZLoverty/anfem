#!/bin/bash

# Define environment name and YAML file
ENV_NAME="ansim"
ENV_FILE="environment.yaml"
REQUIREMENTS="requirements.txt"

# Check if the conda environment exists
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "🛠 Creating conda environment: $ENV_NAME"
    conda env create -f "$ENV_FILE"
else
    echo "✅ Conda environment '$ENV_NAME' already exists."
fi

# Activate the environment and run pip install
echo "🚀 Activating $ENV_NAME and installing pip packages..."
conda run -n "$ENV_NAME" pip install -r "$REQUIREMENTS"

echo "🎉 Done!"
