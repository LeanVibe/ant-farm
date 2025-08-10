#!/bin/bash

# LeanVibe Agent Hive 2.0 - Cleanup Script
# Removes obsolete files identified during technical debt assessment

echo "🧹 LeanVibe Agent Hive 2.0 - System Cleanup"
echo "============================================="

# Safety check - ensure we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src" ]]; then
    echo "❌ Error: Not in the project root directory"
    echo "Please run this script from the ant-farm project root"
    exit 1
fi

echo "📋 Cleanup Report"
echo "------------------"

# Count files before cleanup
echo "🔍 Scanning for obsolete files..."

LOG_COUNT=$(find . -name "*.log" | wc -l | tr -d ' ')
PYC_COUNT=$(find . -name "*.pyc" | wc -l | tr -d ' ')
PYCACHE_COUNT=$(find . -name "__pycache__" -type d | wc -l | tr -d ' ')
DS_STORE_COUNT=$(find . -name ".DS_Store" | wc -l | tr -d ' ')

echo "📊 Files found:"
echo "   - Log files: $LOG_COUNT"
echo "   - Python cache files (.pyc): $PYC_COUNT" 
echo "   - Python cache directories (__pycache__): $PYCACHE_COUNT"
echo "   - System files (.DS_Store): $DS_STORE_COUNT"

TOTAL_FILES=$((LOG_COUNT + PYC_COUNT + PYCACHE_COUNT + DS_STORE_COUNT))
echo "   - Total obsolete files: $TOTAL_FILES"

if [[ $TOTAL_FILES -eq 0 ]]; then
    echo "✅ No obsolete files found - system is clean!"
    exit 0
fi

echo ""
echo "🗑️  Cleaning up obsolete files..."

# Remove log files (keeping recent ones)
if [[ $LOG_COUNT -gt 0 ]]; then
    echo "🧹 Removing old log files..."
    find . -name "*.log" -mtime +1 -delete 2>/dev/null || true
    REMAINING_LOGS=$(find . -name "*.log" | wc -l | tr -d ' ')
    echo "   - Removed $((LOG_COUNT - REMAINING_LOGS)) log files"
    if [[ $REMAINING_LOGS -gt 0 ]]; then
        echo "   - Kept $REMAINING_LOGS recent log files"
    fi
fi

# Remove Python cache files
if [[ $PYC_COUNT -gt 0 ]]; then
    echo "🧹 Removing Python cache files (.pyc)..."
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "   - Removed $PYC_COUNT .pyc files"
fi

# Remove Python cache directories
if [[ $PYCACHE_COUNT -gt 0 ]]; then
    echo "🧹 Removing Python cache directories (__pycache__)..."
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "   - Removed $PYCACHE_COUNT __pycache__ directories"
fi

# Remove system files
if [[ $DS_STORE_COUNT -gt 0 ]]; then
    echo "🧹 Removing system files (.DS_Store)..."
    find . -name ".DS_Store" -delete 2>/dev/null || true
    echo "   - Removed $DS_STORE_COUNT .DS_Store files"
fi

echo ""
echo "📊 Cleanup Summary"
echo "------------------"

# Count remaining files
REMAINING_LOG=$(find . -name "*.log" | wc -l | tr -d ' ')
REMAINING_PYC=$(find . -name "*.pyc" | wc -l | tr -d ' ')
REMAINING_PYCACHE=$(find . -name "__pycache__" -type d | wc -l | tr -d ' ')
REMAINING_DS_STORE=$(find . -name ".DS_Store" | wc -l | tr -d ' ')

REMAINING_TOTAL=$((REMAINING_LOG + REMAINING_PYC + REMAINING_PYCACHE + REMAINING_DS_STORE))
CLEANED_TOTAL=$((TOTAL_FILES - REMAINING_TOTAL))

echo "✅ Cleaned up: $CLEANED_TOTAL files"
echo "📁 Remaining:  $REMAINING_TOTAL files"

if [[ $REMAINING_TOTAL -eq 0 ]]; then
    echo "🎉 Repository is now completely clean!"
else
    echo "ℹ️  Remaining files are recent or necessary"
fi

echo ""
echo "📝 Recommendations"
echo "------------------"
echo "• Run 'git status' to see if any tracked files were affected"
echo "• Consider adding these patterns to .gitignore:"
echo "  - *.log"
echo "  - *.pyc" 
echo "  - __pycache__/"
echo "  - .DS_Store"
echo ""
echo "✅ Cleanup completed successfully!"