#!/bin/bash

# Exit on error
set -e

# Check if we're in a Git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Error: Not in a Git repository. Exiting."
  exit 1
fi

echo "Starting cleanup of unnecessary files..."

# Remove untracked files and directories listed in .gitignore
if [ -f .gitignore ]; then
  echo "Removing untracked files listed in .gitignore..."
  git clean -fdX
else
  echo "No .gitignore file found. Skipping git clean -X."
fi

# Remove common temporary files and directories not necessarily in .gitignore
echo "Removing common temporary files and directories..."
find . -type f -name "*.bak" -delete
find . -type f -name "*.tmp" -delete
find . -type f -name "*.log" -delete
find . -type f -name "*~" -delete
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "node_modules" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name "build" -exec rm -rf {} +
find . -type d -name "dist" -exec rm -rf {} +

# Optionally remove empty directories
echo "Removing empty directories..."
find . -type d -empty -delete

# Check if there are untracked files left
if git status --porcelain | grep -q "^??"; then
  echo "Warning: There are still untracked files in the repository."
  git status --short
else
  echo "No untracked files remain."
fi

echo "Cleanup completed."