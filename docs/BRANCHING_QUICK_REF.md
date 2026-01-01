# Branch Management Quick Reference

Quick guide for managing branches in the Kinetra repository.

## First-Time Setup

```bash
# Set up local main branch tracking remote
make branch-setup
```

## Daily Workflow

### Check Status
```bash
make branch-status
```

### Start New Work
```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature
```

### Save Work
```bash
# Commit changes
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/my-feature
```

### Sync with Remote
```bash
make branch-sync
```

### Finish Work
```bash
# Create PR on GitHub
# After merge, clean up local branch
git checkout main
git pull origin main
git branch -d feature/my-feature
```

## Common Commands

| Task | Command |
|------|---------|
| Check status | `make branch-status` |
| Setup main branch | `make branch-setup` |
| Sync current branch | `make branch-sync` |
| List all branches | `python scripts/branch_manager.py --list --all` |
| Clean merged branches | `python scripts/branch_manager.py --cleanup` |

## Branch Types

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/energy-model` |
| `fix/` | Bug fixes | `fix/nan-handling` |
| `refactor/` | Code refactoring | `refactor/risk-module` |
| `docs/` | Documentation | `docs/api-reference` |
| `test/` | Tests | `test/physics-coverage` |
| `perf/` | Performance | `perf/backtest-speed` |
| `security/` | Security fixes | `security/credential-storage` |

## Troubleshooting

### "Your branch is behind 'origin/main'"
```bash
git checkout main
git pull origin main
```

### "Cannot push to protected branch"
Create a Pull Request instead. Direct pushes to `main` are not allowed.

### "Your branch has diverged"
```bash
# Option 1: Rebase (recommended)
git rebase origin/main

# Option 2: Reset to remote (discards local changes)
git reset --hard origin/main
```

## Full Documentation

See [BRANCHING_STRATEGY.md](BRANCHING_STRATEGY.md) for complete details.
