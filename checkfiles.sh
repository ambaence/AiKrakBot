#!/bin/bash

# Script to check project files and create a requirements.txt for AiKrakBot

# Variables (customize these)
PROJECT_DIR="$HOME/AiKrakBot"  # Path to your project directory
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

# Step 1: Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory $PROJECT_DIR does not exist. Creating it..."
    mkdir -p "$PROJECT_DIR"
else
    echo "Project directory found: $PROJECT_DIR"
fi

cd "$PROJECT_DIR" || { echo "Failed to enter $PROJECT_DIR"; exit 1; }

# Step 2: List all files in the project
echo "Listing all files in $PROJECT_DIR:"
ls -laR  # Recursive listing with details
echo "-------------------------------------"

# Step 3: Check for Python environment and generate requirements.txt
echo "Checking for Python environment to generate requirements.txt..."

# Function to detect active Conda environment
check_conda() {
    if [ -n "$CONDA_DEFAULT_ENV" ] && command -v conda &> /dev/null; then
        echo "Active Conda environment detected: $CONDA_DEFAULT_ENV"
        return 0
    else
        return 1
    fi
}

# Function to detect virtualenv
check_virtualenv() {
    if [ -n "$VIRTUAL_ENV" ] && command -v python &> /dev/null; then
        echo "Active virtualenv detected: $VIRTUAL_ENV"
        return 0
    else
        return 1
    fi
}

# Generate requirements.txt based on environment
if check_conda || check_virtualenv; then
    echo "Generating requirements.txt from current environment..."
    pip freeze > "$REQUIREMENTS_FILE"
    if [ $? -eq 0 ]; then
        echo "requirements.txt created successfully at $REQUIREMENTS_FILE:"
        cat "$REQUIREMENTS_FILE"
    else
        echo "Failed to generate requirements.txt with pip freeze."
        exit 1
    fi
else
    echo "No active Conda or virtualenv detected."
    echo "Creating a basic requirements.txt with common AI/ML packages."
    cat <<EOL > "$REQUIREMENTS_FILE"
# requirements.txt for AiKrakBot
tensorflow-cpu==2.15.0
torch==2.2.1
scikit-learn==1.4.1
pandas==2.2.1
numpy==1.26.4
EOL
    echo "Basic requirements.txt created at $REQUIREMENTS_FILE:"
    cat "$REQUIREMENTS_FILE"
    echo "Edit $REQUIREMENTS_FILE manually to match your project's needs."
fi

# Step 4: Final message
echo "-------------------------------------"
echo "Script complete! Check $PROJECT_DIR for files and $REQUIREMENTS_FILE."
