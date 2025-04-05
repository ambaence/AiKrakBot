#!/bin/bash

PROJECT_DIR="$HOME/Downloads/AiKrakBot"
GITHUB_USERNAME="ambaence"
REPO_NAME="AiKrakBot"
README_FILE="$PROJECT_DIR/README.md"
TREE_FILE="$PROJECT_DIR/file_tree.txt"

if ! command -v tree &> /dev/null; then
    echo "'tree' is not installed. Installing it now..."
    sudo apt update && sudo apt install -y tree
else
    echo "'tree' is already installed."
fi

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating project directory: $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
else
    echo "Project directory already exists: $PROJECT_DIR"
fi

cd "$PROJECT_DIR" || { echo "Failed to enter $PROJECT_DIR"; exit 1; }

echo "Generating file tree..."
tree -a -o "$TREE_FILE" --noreport
echo "File tree saved to $TREE_FILE"

if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
else
    echo "Directory is already a Git repository."
fi

if [ ! -f "$README_FILE" ]; then
    echo "Creating README.md..."
    cat <<EOL > "$README_FILE"
# AiKrakBot
An AI-powered project.

## File Structure
\`\`\`
$(cat "$TREE_FILE")
\`\`\`
EOL
    echo "README.md created."
else
    echo "README.md already exists, skipping creation."
fi

echo "Staging files..."
git add .
git commit -m "Initial commit: Set up AiKrakBot project structure and file tree"

REPO_URL="git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
if ! git ls-remote "$REPO_URL" &> /dev/null; then
    echo "Creating GitHub repository: $REPO_NAME"
    curl -u "$GITHUB_USERNAME" https://api.github.com/user/repos -d "{\"name\":\"$REPO_NAME\"}"
else
    echo "Repository already exists on GitHub."
    echo "Pulling remote changes to avoid conflicts..."
    git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
    git fetch origin
    git pull origin main --allow-unrelated-histories  # Merge unrelated histories if needed
fi

echo "Setting up remote and pushing to GitHub..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
git branch -M main
git push -u origin main

echo "Setup complete! Your AiKrakBot project is now synced with GitHub."
