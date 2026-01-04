#!/bin/bash
# Prune Merged and Stale Branches
# Version: 1.0.0

set -e

echo "🌿 Branch Cleanup - Kinetra"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ensure we're on main
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "⚠️  Not on main branch (current: $current_branch)"
    echo "Switching to main..."
    git checkout main
fi

# Update main from remote
echo ""
echo "📡 Fetching from remote..."
git fetch --all --prune

# Show current branches
echo ""
echo "📊 Current branches:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git branch -a

# List local branches (excluding main)
echo ""
echo "🔍 Analyzing local branches..."
local_branches=$(git branch | grep -v "^\*" | grep -v "main" | sed 's/^[ \t]*//')

if [ -z "$local_branches" ]; then
    echo "✅ No local branches to clean up (only main exists)"
else
    echo "Found local branches:"
    echo "$local_branches" | while read branch; do
        echo "  - $branch"
    done
    
    echo ""
    read -p "Delete these local branches? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$local_branches" | while read branch; do
            echo "🗑️  Deleting: $branch"
            git branch -D "$branch"
        done
        echo "✅ Local branches cleaned"
    else
        echo "⏭️  Skipped local branch cleanup"
    fi
fi

# Check remote branches
echo ""
echo "🔍 Checking remote branches..."
remote_branches=$(git branch -r | grep -v "HEAD" | grep -v "origin/main" | sed 's/^[ \t]*origin\///')

if [ -z "$remote_branches" ]; then
    echo "✅ No remote branches to review (only main exists)"
else
    echo "Found remote branches:"
    echo "$remote_branches" | while read branch; do
        # Check if merged to main
        merged=$(git branch -r --merged origin/main | grep "origin/$branch" || echo "")
        if [ -n "$merged" ]; then
            echo "  - $branch (MERGED)"
        else
            echo "  - $branch (ACTIVE)"
        fi
    done
    
    echo ""
    echo "⚠️  Remote branch cleanup requires manual review"
    echo "Run: git push origin --delete <branch_name>"
fi

# Cleanup tracking references
echo ""
echo "🧹 Cleaning up tracking references..."
git remote prune origin

echo ""
echo "✅ Branch cleanup complete!"
echo ""
echo "📊 Final branch status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git branch -a
