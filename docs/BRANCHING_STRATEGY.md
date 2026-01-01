# Kinetra Branching Strategy

## Overview

This document defines the branching strategy for the Kinetra project, establishing clear workflows for local development, feature branches, and production releases.

## Branch Structure

### Main Branches

#### `main`
- **Purpose**: Production-ready code
- **Protection**: Protected branch, requires PR approval and passing CI
- **Merge Strategy**: Squash and merge from feature branches
- **Lifetime**: Permanent

### Supporting Branches

#### Feature Branches
- **Naming Convention**: `<type>/<description>`
- **Types**:
  - `feature/` - New features or enhancements
  - `fix/` - Bug fixes
  - `refactor/` - Code refactoring without functional changes
  - `docs/` - Documentation updates
  - `test/` - Test additions or improvements
  - `perf/` - Performance improvements
  - `security/` - Security fixes or improvements
- **Source Branch**: `main`
- **Target Branch**: `main`
- **Lifetime**: Temporary (deleted after merge)

#### AI Agent Branches
- **Naming Convention**: `<agent>/<description>`
- **Agents**: `copilot/`, `claude/`, `cursor/`
- **Purpose**: Work-in-progress branches created by AI assistants
- **Cleanup**: Should be merged or deleted after completion

## Workflows

### 1. Setting Up Your Local Repository

```bash
# Clone the repository
git clone https://github.com/sandman9988/Kinetra.git
cd Kinetra

# Set up tracking branch for main
git checkout -b main origin/main

# Verify setup
git branch -vv
```

### 2. Starting New Work

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/new-physics-model

# Make your changes
# ...
```

### 3. Development Workflow

```bash
# Stage and commit changes
git add <files>
git commit -m "Add energy-based regime detection"

# Keep branch updated with main (if needed)
git fetch origin
git rebase origin/main

# Push to remote
git push origin feature/new-physics-model
```

### 4. Creating a Pull Request

1. Push your feature branch to GitHub
2. Create a Pull Request targeting `main`
3. Ensure PR description includes:
   - Purpose of changes
   - Testing performed
   - Any breaking changes
   - Related issue numbers
4. Wait for CI checks to pass
5. Request review if needed
6. Merge when approved and CI passes

### 5. Syncing with Remote

```bash
# Fetch all remote changes
git fetch origin

# Update your main branch
git checkout main
git pull origin main

# Update feature branch with latest main
git checkout feature/your-branch
git rebase origin/main
```

## Best Practices

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `perf`, `chore`

**Examples**:
```
feat(physics): add energy-based regime detection

Implement kinetic energy calculation using price momentum
to identify market regimes (underdamped, critical, overdamped).

Closes #42
```

```
fix(risk): correct RoR calculation for negative equity

The risk-of-ruin formula was not handling edge case where
equity approaches zero. Added bounds checking.
```

### Branch Hygiene

1. **Keep branches short-lived**: Aim to merge within 1-2 weeks
2. **Small, focused changes**: One feature or fix per branch
3. **Regular rebasing**: Keep feature branches updated with main
4. **Delete after merge**: Clean up merged branches
5. **Descriptive names**: Use clear, concise branch names

### Pull Request Guidelines

1. **One concern per PR**: Don't mix features, fixes, and refactors
2. **Include tests**: All new code should have tests
3. **Update documentation**: Keep docs in sync with code
4. **Pass all checks**: Ensure CI passes before requesting review
5. **Link issues**: Reference related issues in PR description

## Branch Protection Rules

### Main Branch

- ✅ Require pull request reviews (1 reviewer)
- ✅ Require status checks to pass
  - Unit tests
  - Integration tests
  - Linting (Ruff)
  - Code formatting (Black)
  - Security scan (CodeQL)
- ✅ Require branches to be up to date
- ✅ Require conversation resolution
- ❌ Allow force pushes (never)
- ❌ Allow deletions (never)

## Common Scenarios

### Scenario 1: Resolving Merge Conflicts

```bash
# Update your branch with latest main
git fetch origin
git checkout feature/your-branch
git rebase origin/main

# If conflicts occur
# 1. Resolve conflicts in files
# 2. Stage resolved files
git add <resolved-files>
git rebase --continue

# Force push (only to your feature branch!)
git push origin feature/your-branch --force-with-lease
```

### Scenario 2: Undoing Changes

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo uncommitted changes
git checkout -- <file>
git restore <file>  # Git 2.23+
```

### Scenario 3: Switching Branches with Uncommitted Changes

```bash
# Option 1: Stash changes
git stash
git checkout other-branch
# Later...
git checkout original-branch
git stash pop

# Option 2: Commit to temporary branch
git checkout -b temp-work
git commit -am "WIP: temporary work"
git checkout other-branch
```

### Scenario 4: Cleaning Up Old Branches

```bash
# List merged branches
git branch --merged main

# Delete local merged branches
git branch -d feature/old-branch

# Delete remote branch
git push origin --delete feature/old-branch

# Prune remote-tracking branches that no longer exist
git fetch origin --prune
```

## Git Configuration Recommendations

Add to your `~/.gitconfig`:

```ini
[user]
    name = Your Name
    email = your.email@example.com

[core]
    editor = vim  # or your preferred editor
    autocrlf = input  # Unix-style line endings

[pull]
    rebase = true  # Rebase instead of merge when pulling

[push]
    default = current  # Push current branch to same-named remote

[rebase]
    autoStash = true  # Automatically stash/unstash when rebasing

[alias]
    # Shortcuts
    co = checkout
    br = branch
    ci = commit
    st = status
    
    # Useful commands
    lg = log --graph --oneline --decorate --all
    unstage = reset HEAD --
    last = log -1 HEAD
    
    # Branch cleanup
    gone = ! "git fetch -p && git for-each-ref --format '%(refname:short) %(upstream:track)' | awk '$2 == \"[gone]\" {print $1}' | xargs -r git branch -D"
```

## Automation

### Git Hooks

The repository includes pre-commit hooks for:
- Code formatting (Black)
- Linting (Ruff)
- Secret detection
- Test execution (on request)

Install hooks:
```bash
# Hooks are managed by pre-commit (if configured)
pre-commit install
```

### CI/CD Integration

GitHub Actions automatically:
1. Runs tests on all PRs
2. Checks code quality (linting, formatting)
3. Performs security scans
4. Runs Monte Carlo backtests
5. Validates theorem proofs

## Troubleshooting

### Issue: "Your branch is behind 'origin/main'"

```bash
git checkout main
git pull origin main
```

### Issue: "Your branch has diverged from 'origin/main'"

```bash
# If you want to keep your changes
git rebase origin/main

# If you want to discard local changes
git reset --hard origin/main
```

### Issue: "Cannot push to protected branch"

You must create a Pull Request. Direct pushes to `main` are not allowed.

### Issue: "Failed to push some refs"

```bash
# Someone else pushed changes
git pull --rebase origin main
git push origin main
```

## References

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Kinetra Contributing Guide](../AI_AGENT_INSTRUCTIONS.md)

---

**Remember**: 
- `main` is always production-ready
- Feature branches are short-lived
- Always create PRs for changes
- Keep commits atomic and well-described
- Test before pushing
