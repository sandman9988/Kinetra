#!/bin/bash
# Prepare GitHub Sync
# Version: 1.0.0

set -e

echo "🔄 GitHub Sync Preparation - Kinetra"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check git status
echo ""
echo "📊 Current Git Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git status

# Check if we're ahead of origin
commits_ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")
echo ""
echo "📈 Commits ahead of origin/main: $commits_ahead"

if [ "$commits_ahead" -gt 0 ]; then
    echo ""
    echo "📝 Recent commits to push:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    git log --oneline origin/main..HEAD
fi

# Check for uncommitted changes
uncommitted=$(git status --porcelain)
if [ -n "$uncommitted" ]; then
    echo ""
    echo "⚠️  Uncommitted changes detected:"
    echo "$uncommitted" | head -20
    echo ""
    read -p "Stage and commit all changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Staging changes..."
        git add -A
        echo ""
        echo "📝 Commit message (or press Enter for default):"
        read -r commit_msg
        if [ -z "$commit_msg" ]; then
            commit_msg="chore: cleanup root docs and prepare for sync

- Archived session-specific status reports
- Consolidated documentation structure
- Added PROJECT_STATUS.md with current state
- Ready for GitHub sync"
        fi
        git commit -m "$commit_msg"
        echo "✅ Changes committed"
    fi
fi

# Final status check
echo ""
echo "📊 Final Status Before Push:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git status

# Calculate what will be pushed
commits_to_push=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")
echo ""
echo "📤 Ready to push $commits_to_push commits to origin/main"

if [ "$commits_to_push" -gt 0 ]; then
    echo ""
    echo "🚀 Push to GitHub with:"
    echo "   git push origin main"
    echo ""
    read -p "Push now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Pushing to GitHub..."
        git push origin main
        echo "✅ Push complete!"
    else
        echo "⏭️  Push skipped - run 'git push origin main' when ready"
    fi
else
    echo "✅ Already in sync with origin/main"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ GitHub sync preparation complete!"
