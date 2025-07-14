#!/bin/bash

set -e

echo "🧹 AUTONOMOUS VISIBILITY PLATFORM - REPOSITORY CLEANUP"
echo "🔍 Scanning and optimizing repository structure..."

PROJECT_DIR="autonomous-visibility-platform"
BACKUP_DIR="repo_backup_$(date +%Y%m%d_%H%M%S)"
COMPRESSED_DIR="visibility_platform_compressed"

# Create backup before cleanup
echo "💾 Creating backup..."
cp -r "$PROJECT_DIR" "$BACKUP_DIR"
echo "✅ Backup created: $BACKUP_DIR"

# Navigate to project directory
cd "$PROJECT_DIR"

echo ""
echo "📊 ANALYZING REPOSITORY STRUCTURE..."

# Function to get directory size
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1 || echo "0B"
}

# Function to count files
count_files() {
    find "$1" -type f 2>/dev/null | wc -l || echo "0"
}

# Scan each level and report
echo ""
echo "📁 CURRENT REPOSITORY ANALYSIS:"
echo "├── Root: $(get_size .) ($(count_files .) files)"

for dir in */; do
    if [ -d "$dir" ]; then
        echo "├── $dir: $(get_size "$dir") ($(count_files "$dir") files)"
        
        # Show subdirectories
        for subdir in "$dir"*/; do
            if [ -d "$subdir" ]; then
                echo "│   ├── $subdir: $(get_size "$subdir") ($(count_files "$subdir") files)"
            fi
        done
    fi
done

echo ""
echo "🗑️ REMOVING UNNECESSARY FILES..."

# Remove common unnecessary files and directories
echo "Removing development artifacts..."

# Node.js artifacts
if [ -d "frontend/node_modules" ]; then
    echo "  🗑️ Removing node_modules ($(get_size frontend/node_modules))"
    rm -rf frontend/node_modules
fi

if [ -f "frontend/package-lock.json" ]; then
    echo "  🗑️ Removing package-lock.json"
    rm -f frontend/package-lock.json
fi

# Python artifacts
echo "  🗑️ Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -exec rm -f {} + 2>/dev/null || true
find . -type f -name "*.pyo" -exec rm -f {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Virtual environments
if [ -d "backend/venv" ]; then
    echo "  🗑️ Removing Python virtual environment ($(get_size backend/venv))"
    rm -rf backend/venv
fi

if [ -d "backend/env" ]; then
    echo "  🗑️ Removing Python environment ($(get_size backend/env))"
    rm -rf backend/env
fi

# Build artifacts
echo "  🗑️ Removing build artifacts..."
rm -rf frontend/.next 2>/dev/null || true
rm -rf frontend/out 2>/dev/null || true
rm -rf frontend/build 2>/dev/null || true
rm -rf frontend/dist 2>/dev/null || true

# Database files
echo "  🗑️ Removing development databases..."
rm -rf data/duckdb/*.db 2>/dev/null || true
rm -rf data/duckdb/*.db-wal 2>/dev/null || true
rm -rf data/duckdb/*.db-shm 2>/dev/null || true

# Log files
echo "  🗑️ Removing log files..."
rm -rf logs/*.log 2>/dev/null || true
rm -rf backend/logs/*.log 2>/dev/null || true

# Temporary files
echo "  🗑️ Removing temporary files..."
find . -type f -name "*.tmp" -exec rm -f {} + 2>/dev/null || true
find . -type f -name "*.temp" -exec rm -f {} + 2>/dev/null || true
find . -type f -name ".DS_Store" -exec rm -f {} + 2>/dev/null || true
find . -type f -name "Thumbs.db" -exec rm -f {} + 2>/dev/null || true

# IDE files
echo "  🗑️ Removing IDE artifacts..."
rm -rf .vscode 2>/dev/null || true
rm -rf .idea 2>/dev/null || true
rm -rf *.swp *.swo 2>/dev/null || true

# ML model artifacts (keep structure but remove large trained models)
if [ -d "ml_engine/models/trained" ]; then
    echo "  🗑️ Removing large trained models..."
    rm -rf ml_engine/models/trained/* 2>/dev/null || true
fi

if [ -d "ml_engine/data/cache" ]; then
    echo "  🗑️ Removing ML cache..."
    rm -rf ml_engine/data/cache/* 2>/dev/null || true
fi

echo ""
echo "📦 COMPRESSING ESSENTIAL FILES..."

# Create compressed version directory
cd ..
mkdir -p "$COMPRESSED_DIR"

# Copy essential files with compression
echo "📋 Copying core application files..."

# Copy main structure (excluding the files we just removed)
rsync -av --exclude='node_modules' \
          --exclude='venv' \
          --exclude='env' \
          --exclude='.next' \
          --exclude='build' \
          --exclude='dist' \
          --exclude='*.pyc' \
          --exclude='__pycache__' \
          --exclude='*.log' \
          --exclude='.DS_Store' \
          --exclude='Thumbs.db' \
          --exclude='.vscode' \
          --exclude='.idea' \
          "$PROJECT_DIR/" "$COMPRESSED_DIR/"

echo ""
echo "🗜️ CREATING COMPRESSED ARCHIVES..."

# Create different compression formats
echo "Creating zip archive..."
zip -r "${COMPRESSED_DIR}.zip" "$COMPRESSED_DIR" > /dev/null 2>&1

echo "Creating tar.gz archive..."
tar -czf "${COMPRESSED_DIR}.tar.gz" "$COMPRESSED_DIR" > /dev/null 2>&1

echo "Creating tar.bz2 archive (highest compression)..."
tar -cjf "${COMPRESSED_DIR}.tar.bz2" "$COMPRESSED_DIR" > /dev/null 2>&1

echo ""
echo "📊 CLEANUP RESULTS:"
echo "├── Original size: $(get_size "$PROJECT_DIR")"
echo "├── Cleaned size: $(get_size "$COMPRESSED_DIR")"
echo "├── Zip archive: $(get_size "${COMPRESSED_DIR}.zip")"
echo "├── Tar.gz archive: $(get_size "${COMPRESSED_DIR}.tar.gz")" 
echo "└── Tar.bz2 archive: $(get_size "${COMPRESSED_DIR}.tar.bz2")"

echo ""
echo "📁 FINAL STRUCTURE ANALYSIS:"
echo "├── Cleaned directory: $COMPRESSED_DIR"
echo "├── Backup directory: $BACKUP_DIR"
echo "└── Compressed archives:"
echo "    ├── ${COMPRESSED_DIR}.zip"
echo "    ├── ${COMPRESSED_DIR}.tar.gz"
echo "    └── ${COMPRESSED_DIR}.tar.bz2"

echo ""
echo "🧹 DETAILED CLEANUP REPORT:"

# Generate detailed report
REPORT_FILE="cleanup_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$REPORT_FILE" << EOF
AUTONOMOUS VISIBILITY PLATFORM - CLEANUP REPORT
Generated: $(date)

REMOVED FILES/DIRECTORIES:
├── Node.js artifacts (node_modules, package-lock.json)
├── Python cache (__pycache__, *.pyc, *.pyo, *.egg-info)
├── Virtual environments (venv, env)
├── Build artifacts (.next, out, build, dist)
├── Development databases (*.db, *.db-wal, *.db-shm)
├── Log files (*.log)
├── Temporary files (*.tmp, *.temp, .DS_Store, Thumbs.db)
├── IDE artifacts (.vscode, .idea, *.swp, *.swo)
├── Large trained ML models
└── ML cache files

PRESERVED STRUCTURE:
├── Source code (frontend/, backend/, ml_engine/)
├── Configuration files (package.json, requirements.txt, etc.)
├── Documentation (README files)
├── Infrastructure configs (docker/, k8s/)
├── Essential data schemas
└── Core application logic

COMPRESSION RESULTS:
├── Original: $(get_size "$PROJECT_DIR")
├── Cleaned: $(get_size "$COMPRESSED_DIR")
├── Space saved: $(echo "$(du -sk "$PROJECT_DIR" | cut -f1) - $(du -sk "$COMPRESSED_DIR" | cut -f1)" | bc 2>/dev/null || echo "N/A")KB
└── Compression ratio: $(echo "scale=1; $(du -sk "$COMPRESSED_DIR" | cut -f1) * 100 / $(du -sk "$PROJECT_DIR" | cut -f1)" | bc 2>/dev/null || echo "N/A")%

FILES RETAINED FOR DEVELOPMENT:
├── All source code (.tsx, .py, .js, .ts files)
├── Configuration files (.json, .env.example, .yml)
├── Documentation (.md files)
├── Shell scripts (.sh files)
└── Essential data files

RECOMMENDED NEXT STEPS:
1. Test the compressed version: cd $COMPRESSED_DIR && npm install
2. Deploy from: ${COMPRESSED_DIR}.tar.gz (best balance of size/compatibility)
3. Archive backup: $BACKUP_DIR (keep for rollback if needed)
4. Development setup: Use cleaned directory for faster operations

EOF

echo "📝 Detailed report saved: $REPORT_FILE"

echo ""
echo "✅ CLEANUP COMPLETED SUCCESSFULLY!"
echo ""
echo "🚀 NEXT STEPS:"
echo "1. Test cleaned version: cd $COMPRESSED_DIR"
echo "2. Deploy compressed archive: ${COMPRESSED_DIR}.tar.gz"
echo "3. Keep backup for safety: $BACKUP_DIR"
echo ""
echo "🎯 Repository is now optimized for production deployment!"