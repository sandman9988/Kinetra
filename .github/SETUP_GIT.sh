#!/bin/bash
# Setup Git configuration for Kinetra project

set -e

echo "🔧 Setting up Git configuration for Kinetra..."
echo ""

# 1. Include project-specific git config
echo "📝 Adding Kinetra git aliases..."
if ! grep -q ".gitconfig-kinetra" ~/.gitconfig 2>/dev/null; then
    echo "" >> ~/.gitconfig
    echo "# Kinetra project configuration" >> ~/.gitconfig
    echo "[include]" >> ~/.gitconfig
    echo "    path = $(pwd)/.gitconfig-kinetra" >> ~/.gitconfig
    echo "✅ Added to ~/.gitconfig"
else
    echo "⚠️  Already included in ~/.gitconfig"
fi

# 2. Setup git hooks
echo ""
echo "🪝 Setting up git hooks..."
git config core.hooksPath .github/hooks
chmod +x .github/hooks/*
echo "✅ Git hooks enabled"

# 3. Configure branch protection (local)
echo ""
echo "🛡️  Configuring branch settings..."
git config branch.autoSetupMerge always
git config branch.autoSetupRebase always
git config pull.rebase true
git config fetch.prune true
git config push.default current
git config push.autoSetupRemote true
git config rebase.autoStash true
echo "✅ Branch settings configured"

# 4. Show available aliases
echo ""
echo "✨ Available Git aliases:"
echo ""
echo "  git cb <ide> \"description\"    - Create IDE branch"
echo "  git clean-merged               - Delete merged branches"
echo "  git clean-merged-safe          - Delete with confirmation"
echo "  git branches-claude            - Show Claude branches"
echo "  git branches-copilot           - Show Copilot branches"
echo "  git branches-cursor            - Show Cursor branches"
echo "  git branches-status            - Show all IDE branches"
echo "  git sync-main                  - Sync current branch with main"
echo "  git aicommit \"message\"         - Commit with AI attribution"
echo "  git tree                       - Show branch tree"
echo "  git info                       - Show current branch info"
echo ""

# 5. Test aliases
echo "🧪 Testing aliases..."
if git cb 2>&1 | grep -q "Usage: git cb"; then
    echo "✅ Aliases working correctly"
else
    echo "⚠️  Aliases may need manual configuration"
fi

echo ""
echo "✅ Git setup complete!"
echo ""
echo "📚 See .github/BRANCHING_STRATEGY.md for usage guidelines"
echo ""
echo "Example usage:"
echo "  git cb claude \"lock production dependencies\""
echo "  # Work on feature..."
echo "  git aicommit \"Lock all dependencies for production\""
echo "  git push"
