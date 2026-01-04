#!/bin/bash
# Consolidation Script - Remove Duplicates, Version Existing Files

set -e

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                    CONSOLIDATION: NO MORE DUPLICATES                      ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Delete duplicate parallel_data_prep.py
echo "1. Removing duplicate parallel_data_prep.py..."
if [ -f "scripts/data/parallel_data_prep.py" ]; then
    rm scripts/data/parallel_data_prep.py
    echo "   ✅ Deleted scripts/data/parallel_data_prep.py (duplicate)"
else
    echo "   ⚠️  Already removed"
fi

# 2. Archive smart_download_menu.py (features will be merged into download_interactive.py)
echo ""
echo "2. Archiving smart_download_menu.py for reference..."
mkdir -p archive/enhanced_versions
if [ -f "scripts/download/smart_download_menu.py" ]; then
    mv scripts/download/smart_download_menu.py archive/enhanced_versions/
    echo "   ✅ Archived to archive/enhanced_versions/"
else
    echo "   ⚠️  Already archived"
fi

# 3. List files that need version numbers
echo ""
echo "3. Files to version (will be enhanced manually):"
echo "   - scripts/download/download_interactive.py → v2.0.0"
echo "   - scripts/download/parallel_data_prep.py → v2.0.0"
echo "   - scripts/download/metaapi_bulk_download.py → v2.0.0"
echo ""

# 4. List new modules (NOT duplicates)
echo "4. New modules (keep these):"
echo "   ✅ kinetra/cpu_utils.py (new utility)"
echo "   ✅ kinetra/menu_state_tracker.py (new utility)"
echo "   ✅ kinetra/exploration_lab/ (new feature)"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                         CONSOLIDATION COMPLETE                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next Steps:"
echo "  1. Manually enhance existing files (don't create new ones)"
echo "  2. Add __version__ = \"2.0.0\" to enhanced files"
echo "  3. Add CHANGELOG section in docstrings"
echo ""

