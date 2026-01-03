# Branch Cleanup Plan

**Date**: 2026-01-03  
**Repository**: sandman9988/Kinetra  
**Total Branches**: 19 (excluding main)  
**Action Required**: Clean up 17 stale branches

---

## Summary

After analysis, we have:
- **4 fully merged branches** - Safe to delete immediately
- **11 branches with minimal unique commits** - Review and likely delete
- **2 branches with significant work** - Review carefully before deleting

---

## Category 1: Fully Merged (Safe to Delete - 4 branches)

These branches have been completely merged into main and have no unique commits:

1. ✅ `claude/add-market-types-HikA5`
2. ✅ `claude/review-changes-mjsnymnrq0wocc5o-7c5RU`
3. ✅ `claude/review-changes-mjtc82dl15hpr98o-PO09N`
4. ✅ `claude/review-changes-mjtutqrh8790iaq8-78lE1`

**Action**: Delete immediately

---

## Category 2: Single Commit Not Merged (Review - 10 branches)

These branches have only 1 unique commit ahead of main. Need to check if valuable:

### claude/metaapi-ohlc-candles-S0SvB
- **Unique Commits**: 1
- **Last Commit**: `77b61f5` - "Add MetaAPI data download and sync system"
- **Assessment**: ⚠️ **POTENTIALLY VALUABLE** - MetaAPI functionality
- **Recommendation**: Review commit, cherry-pick if needed, then delete branch

### cursor/codebase-performance-optimization-5c6a
- **Unique Commits**: 1
- **Assessment**: Likely superseded by later work
- **Recommendation**: Review and delete

### cursor/codebase-performance-optimization-5f8e
- **Unique Commits**: 1
- **Assessment**: Likely superseded by later work
- **Recommendation**: Review and delete

### cursor/codebase-performance-optimization-f557
- **Unique Commits**: 1
- **Assessment**: Likely superseded by later work
- **Recommendation**: Review and delete

### cursor/docker-build-error-handling-0852
- **Unique Commits**: 1
- **Assessment**: Docker fixes likely incorporated elsewhere
- **Recommendation**: Review and delete

### cursor/docker-build-error-handling-8ef7
- **Unique Commits**: 1
- **Assessment**: Docker fixes likely incorporated elsewhere
- **Recommendation**: Review and delete

### cursor/docker-build-error-handling-a7f3
- **Unique Commits**: 1
- **Assessment**: Docker fixes likely incorporated elsewhere
- **Recommendation**: Review and delete

### cursor/docker-build-error-handling-a8c1
- **Unique Commits**: 1
- **Assessment**: Docker fixes likely incorporated elsewhere
- **Recommendation**: Review and delete

### cursor/docker-build-error-handling-ee63
- **Unique Commits**: 1
- **Assessment**: Docker fixes likely incorporated elsewhere
- **Recommendation**: Review and delete

### claude/review-changes-mjrmrrn2004i8yy7-rkv27
- **Unique Commits**: 1 (approximate - needs verification)
- **Assessment**: Review change branch, likely obsolete
- **Recommendation**: Review and delete

---

## Category 3: Multiple Commits Not Merged (Review Carefully - 2 branches)

### cursor/codebase-performance-optimization-8966
- **Unique Commits**: 27 commits
- **Key Commits**:
  - `b9bf299` - Checkpoint before follow-up message
  - `43cff32` - feat: Add pathfinder deep dive script
  - `6e27f55` - Refactor: Add ROCm support and update Dockerfile
  - `e9ffdad` - Refactor physics calculations and agent initializations
  - `4911c66` - Refactor margin and lot size calculations to use symbol specs
- **Assessment**: ⚠️ **SIGNIFICANT WORK** - Contains ROCm support, physics refactoring
- **Recommendation**: 
  1. Review commits carefully
  2. Check if ROCm support was integrated elsewhere
  3. Cherry-pick valuable commits if not integrated
  4. Then delete branch

### cursor/codebase-performance-optimization-d42f
- **Unique Commits**: 6 commits
- **Assessment**: ⚠️ **MODERATE WORK** - Performance optimizations
- **Recommendation**: Review and cherry-pick if needed

---

## Dependabot Branches (Keep - 2 branches)

These are automated dependency updates - keep for reference:

1. ✅ `dependabot/pip/pip-6370cf6e86` - **MERGED** (PR #105)
2. ⏳ `dependabot/pip/pip-8f956fbd8d` - May be pending

**Action**: Check if merged, then dependabot will auto-cleanup

---

## Cleanup Commands

### Step 1: Delete Fully Merged Branches (Safe)
```bash
git push origin --delete claude/add-market-types-HikA5
git push origin --delete claude/review-changes-mjsnymnrq0wocc5o-7c5RU
git push origin --delete claude/review-changes-mjtc82dl15hpr98o-PO09N
git push origin --delete claude/review-changes-mjtutqrh8790iaq8-78lE1
```

### Step 2: Review and Save Important Work

#### Review MetaAPI branch
```bash
git show origin/claude/metaapi-ohlc-candles-S0SvB:kinetra/mt5_connector.py > /tmp/metaapi_review.py
# Review the file, cherry-pick if needed:
git cherry-pick 77b61f5
```

#### Review Performance Optimization (27 commits)
```bash
git log origin/main..origin/cursor/codebase-performance-optimization-8966 --oneline > /tmp/perf_commits.txt
# Review each commit:
git show origin/cursor/codebase-performance-optimization-8966:Dockerfile > /tmp/rocm_review.txt
# If ROCm not in main, consider cherry-picking:
git cherry-pick 6e27f55  # ROCm support
```

#### Review Performance Optimization (6 commits)
```bash
git log origin/main..origin/cursor/codebase-performance-optimization-d42f --oneline
# Cherry-pick valuable commits if any
```

### Step 3: Delete Single-Commit Branches
```bash
# After reviewing, delete these:
git push origin --delete claude/metaapi-ohlc-candles-S0SvB
git push origin --delete claude/review-changes-mjrmrrn2004i8yy7-rkv27
git push origin --delete cursor/codebase-performance-optimization-5c6a
git push origin --delete cursor/codebase-performance-optimization-5f8e
git push origin --delete cursor/codebase-performance-optimization-f557
git push origin --delete cursor/docker-build-error-handling-0852
git push origin --delete cursor/docker-build-error-handling-8ef7
git push origin --delete cursor/docker-build-error-handling-a7f3
git push origin --delete cursor/docker-build-error-handling-a8c1
git push origin --delete cursor/docker-build-error-handling-ee63
```

### Step 4: Delete Multi-Commit Branches (After Review)
```bash
# Only after carefully reviewing and cherry-picking:
git push origin --delete cursor/codebase-performance-optimization-8966
git push origin --delete cursor/codebase-performance-optimization-d42f
```

### Step 5: Bulk Delete (Alternative - After Full Review)
```bash
# Delete all at once (use with caution):
git push origin --delete \
  claude/add-market-types-HikA5 \
  claude/metaapi-ohlc-candles-S0SvB \
  claude/review-changes-mjrmrrn2004i8yy7-rkv27 \
  claude/review-changes-mjsnymnrq0wocc5o-7c5RU \
  claude/review-changes-mjtc82dl15hpr98o-PO09N \
  claude/review-changes-mjtutqrh8790iaq8-78lE1 \
  cursor/codebase-performance-optimization-5c6a \
  cursor/codebase-performance-optimization-5f8e \
  cursor/codebase-performance-optimization-8966 \
  cursor/codebase-performance-optimization-d42f \
  cursor/codebase-performance-optimization-f557 \
  cursor/docker-build-error-handling-0852 \
  cursor/docker-build-error-handling-8ef7 \
  cursor/docker-build-error-handling-a7f3 \
  cursor/docker-build-error-handling-a8c1 \
  cursor/docker-build-error-handling-ee63
```

---

## Pre-Deletion Checklist

Before deleting any branch:

- [ ] Verify branch is not referenced in open PRs
- [ ] Check if branch has unique commits not in main
- [ ] Review commit messages for valuable work
- [ ] Cherry-pick any important commits
- [ ] Document decision in this file
- [ ] Confirm with team if branch name suggests important work

---

## Preservation Strategy

### Create Backup Tag (Optional)
Before deleting branches with significant work, create tags:

```bash
# For the 27-commit performance branch:
git tag archive/perf-optimization-8966 origin/cursor/codebase-performance-optimization-8966
git push origin archive/perf-optimization-8966

# Then delete branch:
git push origin --delete cursor/codebase-performance-optimization-8966
```

---

## Recommended Execution Order

### Phase 1: Safe Cleanup (Do First)
1. ✅ Delete 4 fully-merged branches
2. ✅ Verify in GitHub that branches are gone

### Phase 2: Review Work (Do Second)
1. ⚠️ Review MetaAPI commit (`claude/metaapi-ohlc-candles-S0SvB`)
2. ⚠️ Review 27-commit performance branch (`cursor/codebase-performance-optimization-8966`)
3. ⚠️ Review 6-commit performance branch (`cursor/codebase-performance-optimization-d42f`)
4. ✅ Cherry-pick any valuable commits

### Phase 3: Cleanup Remaining (Do Third)
1. ✅ Delete all single-commit branches (10 branches)
2. ✅ Delete multi-commit branches (after preservation)
3. ✅ Verify final branch count

---

## Expected Final State

After cleanup:
- **main** branch (production)
- **dependabot/** branches (if any active PRs)
- **Total branches**: 1-3 (down from 19)

---

## Risk Assessment

| Risk Level | Description | Mitigation |
|------------|-------------|------------|
| 🟢 **LOW** | Deleting fully merged branches | None needed - safe |
| 🟡 **MEDIUM** | Deleting single-commit branches | Review commit first |
| 🔴 **HIGH** | Deleting 27-commit branch | Tag before delete, thorough review |

---

## Next Steps

1. **Execute Phase 1** (safe cleanup) immediately
2. **Review and cherry-pick** valuable commits from Phase 2
3. **Execute Phase 3** after preservation
4. **Update this document** with results

---

**Status**: 📋 Plan Ready  
**Last Updated**: 2026-01-03  
**Prepared By**: Kinetra Development Team