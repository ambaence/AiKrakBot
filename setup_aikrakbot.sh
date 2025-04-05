#!/bin/bash

# Script to create a file tree and set up a Git repository for AiKrakBot

# Variables (customize these)
PROJECT_DIR="$HOME/Downloads/AiKrakBot"  # Path to your project directory
GITHUB_USERNAME="ambaence"     # Your GitHub username
REPO_NAME="AiKrakBot"          # Desired GitHub repository name
README_FILE="$PROJECT_DIR/README.md"
TREE_FILE="$PROJECT_DIR/file_tree.txt"

# Step 1: Check if 'tree' is installed, install if not
if ! command -v tree &> /dev/null; then
    echo "'tree' is not installed. Installing it now..."
    sudo apt update && sudo apt install -y tree
    if [ $? -ne 0 ]; then
        echo "Failed to install 'tree'. Exiting."
        exit 1
    fi
else
    echo "'tree' is already installed."
fi

# Step 2: Create project directory if it doesn’t exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating project directory: $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
else
    echo "Project directory already exists: $PROJECT_DIR"
fi

cd "$PROJECT_DIR" || { echo "Failed to enter $PROJECT_DIR"; exit 1; }

# Step 3: Generate file tree and save it
echo "Generating file tree..."
tree -a -o "$TREE_FILE" --noreport
if [ $? -eq 0 ]; then
    echo "File tree saved to $TREE_FILE"
else
    echo "Failed to generate file tree. Exiting."
    exit 1
fi

# Step 4: Initialize Git repository if not already a repo
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
else
    echo "Directory is already a Git repository."
fi

# Step 5: Create a basic README if it doesn’t exist
if [ ! -f "$README_FILE" ]; then
    echo "Creating README.md..."
    cat <<EOL > "$README_FILE"
# AiKrakBot
An AI-powered project for [describe your project briefly].

## File Structure
\`\`\`
$(cat "$TREE_FILE")
\`\`\`
EOL
    echo "README.md created."
else
    echo "README.md already exists, skipping creation."
fi

# Step 6: Stage and commit files
echo "Staging files..."
git add .
git commit -m "Initial commit: Set up AiKrakBot project structure and file tree"

# Step 7: Create remote repository on GitHub (if not already created)
echo "Checking if GitHub repository exists..."
REPO_URL="git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
if ! git ls-remote "$REPO_URL" &> /dev/null; then
    echo "Creating GitHub repository: $REPO_NAME"
    curl -u "$GITHUB_USERNAME" https://api.github.com/user/repos -d "{\"name\":\"$REPO_NAME\"}"
    if [ $? -eq 0 ]; then
        echo "Repository created successfully."
    else
        echo "Failed to create repository. Please create it manually on GitHub and try again."
        exit 1
    fi
else
    echo "Repository already exists on GitHub."
fi

# Step 8: Add remote and push
echo "Setting up remote and pushing to GitHub..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo "Successfully pushed to GitHub: $REPO_URL"
else
    echo "Failed to push to GitHub. Check your SSH keys and network connection."
    exit 1
fi

echo "Setup complete! Your AiKrakBot project is now a Git repository on GitHub."
