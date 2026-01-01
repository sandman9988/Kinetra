# Kinetra Branching Strategy

## Overview

**Branch per IDE, not per AI agent.** Multiple AI agents working in the same IDE/terminal share the same branch.

## Branch Naming Convention

```
<ide>/<feature-description>
```

### IDE Prefixes
- `claude/` - Claude Code (this IDE)
- `copilot/` - GitHub Copilot
- `cursor/` - Cursor IDE
- `windsurf/` - Windsurf IDE
- `manual/` - Manual development (no AI)

### Examples
```
claude/production-dependencies-lock
copilot/fix-authentication-flow
cursor/optimize-backtest-performance
manual/hotfix-critical-bug
```

## Workflow

### 1. Starting New Work in an IDE

```bash
# In Claude Code
git checkout -b claude/your-feature-description

# In Copilot
git checkout -b copilot/your-feature-description

# In Cursor
git checkout -b cursor/your-feature-description
```

### 2. Multiple AI Agents in Same IDE

**Different agents in the same terminal/IDE SHARE the same branch.**

Example: Claude Code IDE
- Agent session 1: Works on `claude/production-dependencies-lock`
- Agent session 2 (later): Continues on `claude/production-dependencies-lock`
- Agent session 3 (next day): Still on `claude/production-dependencies-lock`

**Why?** The branch represents the IDE's work, not individual agent sessions.

### 3. Switching IDEs

If you switch IDEs mid-feature:

```bash
# Finish work in Claude Code
git add .
git commit -m "WIP: partial implementation"
git push origin claude/feature-name

# Switch to Cursor, create new branch from current state
git checkout claude/feature-name
git checkout -b cursor/continue-feature-name
```

### 4. Merging to Main

```bash
# Ensure branch is up to date
git checkout claude/your-feature
git pull origin main
git rebase main  # or merge main if you prefer

# Push and create PR
git push origin claude/your-feature

# Create PR on GitHub targeting main
# After approval, merge via GitHub
```

### 5. Cleaning Up Merged Branches

**Local cleanup:**
```bash
# Delete local branches merged to main
git branch --merged main | grep -v "^\*" | grep -v "main" | xargs git branch -d

# Or manually
git branch -d claude/feature-name
```

**Remote cleanup:**
```bash
# Delete remote branch after merge
git push origin --delete claude/feature-name
```

## Branch Lifecycle

```
main (protected)
  │
  ├─→ claude/feature-A ──→ PR ──→ merge ──→ delete
  │
  ├─→ copilot/feature-B ──→ PR ──→ merge ──→ delete
  │
  └─→ cursor/feature-C ──→ PR ──→ merge ──→ delete
```

## Rules

### ✅ DO
- Create one branch per IDE for each feature/fix
- Use descriptive feature names (not agent session IDs)
- Reuse the same branch for multiple agent sessions in the same IDE
- Clean up branches after merge
- Keep branches focused on single features

### ❌ DON'T
- Create new branch for each AI agent conversation
- Use random IDs in branch names (e.g., `claude/review-changes-mjplvql0feekcfx6-2Zjmd`)
- Leave merged branches lingering
- Mix multiple unrelated features in one branch
- Work on main directly

## Example Workflow

```bash
# Day 1: Start feature in Claude Code
git checkout -b claude/dependency-locking
# ... work with AI agent session 1 ...
git commit -m "Add locked dependencies to pyproject.toml"
git push origin claude/dependency-locking

# Day 2: Continue in Claude Code (new agent session)
git checkout claude/dependency-locking
# ... work with AI agent session 2 ...
git commit -m "Add installation documentation"
git push origin claude/dependency-locking

# Day 3: Final touches in Claude Code (another new session)
git checkout claude/dependency-locking
# ... work with AI agent session 3 ...
git commit -m "Lock all dependencies for production"
git push origin claude/dependency-locking

# Create PR and merge
# GitHub PR: claude/dependency-locking → main

# Cleanup after merge
git checkout main
git pull origin main
git branch -d claude/dependency-locking
git push origin --delete claude/dependency-locking
```

## Git Aliases (Optional)

Add to `~/.gitconfig`:

```ini
[alias]
    # Create IDE branch
    cb = "!f() { git checkout -b ${1}/$(echo ${2} | tr ' ' '-'); }; f"

    # Clean merged branches
    clean-merged = "!git branch --merged main | grep -v '^\\*' | grep -v 'main' | xargs -r git branch -d"

    # Show branches by IDE
    branches-claude = "!git branch -a | grep claude"
    branches-copilot = "!git branch -a | grep copilot"
    branches-cursor = "!git branch -a | grep cursor"
```

Usage:
```bash
# Create branch: git cb <ide> <description>
git cb claude "fix authentication"
# Creates: claude/fix-authentication

# Clean up merged branches
git clean-merged

# See all Claude branches
git branches-claude
```

## Why This Strategy?

1. **Clear ownership** - Know which IDE created which changes
2. **Less clutter** - No new branch for every AI conversation
3. **Better tracking** - Feature-focused branches, not session-focused
4. **Easy collaboration** - Multiple sessions on same feature share progress
5. **Simple cleanup** - Delete by feature, not by session

## Migration from Old Branches

Current state has many auto-generated branch names with random IDs. Clean up:

```bash
# List all your branches
git branch -a

# Delete merged local branches
git branch --merged main | grep -v "^\*" | grep -v "main" | xargs git branch -d

# Delete merged remote branches (via GitHub PR interface or)
git push origin --delete <branch-name>
```

## Questions?

- **Q:** What if I start in Claude Code and want to continue in Cursor?
- **A:** Create new branch from current state: `git checkout -b cursor/continue-feature`

- **Q:** Do I create new branch for every AI conversation?
- **A:** No! Reuse the same branch for all conversations in the same IDE working on the same feature.

- **Q:** What about hotfixes?
- **A:** Use your IDE prefix: `claude/hotfix-critical-bug` or `manual/hotfix-prod-crash`

- **Q:** Can I work on main directly?
- **A:** No! Always use branches, even for small changes.
