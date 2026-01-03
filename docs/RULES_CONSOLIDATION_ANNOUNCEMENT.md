# 📢 Canonical Rules System - Team Announcement

**Date:** January 9, 2024  
**Status:** Active  
**Priority:** High - Please Read

---

## 🎯 What Changed?

We've consolidated **all agent rules and development guidelines** into a **single canonical rulebook**.

### Before 😰
- Rules scattered across 8+ files
- Contradictions between sources
- Confusion about which rules to follow
- AI agents getting mixed signals
- Outdated copies in archive folders

### After 🎉
- ✅ **One source of truth:** [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)
- ✅ All 17 rule sections in one place
- ✅ Quick references point to master
- ✅ Automated enforcement via CI
- ✅ No contradictions

---

## 📚 The New Structure

```
AGENT_RULES_MASTER.md (ROOT - Single Source of Truth)
├── .github/copilot-instructions.md (Quick reference → points to master)
├── .claude/instructions.md (Quick reference → points to master)
├── .claude/type_checking_guidelines.md (Quick reference → points to master)
└── archive/*/AI_AGENT_INSTRUCTIONS.md (Deprecated → points to master)
```

---

## 🚀 What You Need to Do

### 1. Read the Master Rulebook (15 minutes)

**Location:** [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)

Skim all 17 sections to understand what's covered:
1. Meta-Rules (Conversation & Compression)
2. Core Philosophy (First Principles)
3. Data Safety & Integrity
4. Performance - Vectorization
5. Memory & Efficiency
6. I/O & Concurrency
7. Determinism & Reproducibility
8. Backtesting Engine
9. Experiment Safety & Validation
10. MetaAPI Connector
11. Logging & Error Handling
12. Security & Hard Prohibitions
13. Code Quality & Style
14. Type Checking & Documentation
15. Physics-First Approach
16. Testing Requirements
17. Deliverables & Validation

### 2. Install Pre-Commit Hook (30 seconds)

```bash
bash .github/SETUP_GIT.sh
```

This will automatically:
- Validate code against rules before commits
- Check for hardcoded credentials
- Check for banned TA indicators
- Backup data directory if needed

### 3. Run Rules Linter (2 minutes)

```bash
# Check your current work
python scripts/lint_rules.py kinetra/

# Check specific files
python scripts/lint_rules.py path/to/your/files.py
```

---

## 🛡️ Automated Enforcement

### Pre-Commit Hook

Every commit now automatically:
- ✅ Validates against canonical rules
- ✅ Checks for hardcoded credentials
- ✅ Checks for traditional TA indicators (RSI, MACD, etc.)
- ✅ Backs up data directory
- ✅ Prevents direct commits to `main`

### CI Pipeline

New `rules-validation` job in CI:
- ✅ Lints all code against rules
- ✅ Checks canonical references exist
- ✅ Scans for security violations
- ✅ Validates data safety patterns
- ✅ Must pass before merge

---

## 🔴 Hard Prohibitions (CI Will Fail)

These are **NEVER** negotiable:

| ❌ Prohibited | ✅ Required Instead |
|--------------|-------------------|
| Traditional TA indicators (RSI, MACD, BB, ATR, ADX) | Physics-based features only |
| Hardcoded credentials | Environment variables (`.env`) |
| Live order placement code | Backtest/paper trading only |
| Direct file writes | `PersistenceManager.atomic_save()` |
| Unseeded RNG in backtests | `np.random.seed()` + `random.seed()` |
| Magic numbers (static thresholds) | Rolling percentiles / DSP-derived |

---

## 📖 Quick Reference Guide

### For Daily Work

Use the quick reference for your tool:
- **GitHub Copilot:** [`.github/copilot-instructions.md`](../.github/copilot-instructions.md)
- **Claude/Zed AI:** [`.claude/instructions.md`](../.claude/instructions.md)
- **Type Checking:** [`.claude/type_checking_guidelines.md`](../.claude/type_checking_guidelines.md)

### For Deep Dives

Check the canonical master:
- **Complete Rules:** [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)

### For Understanding the System

Read the guide:
- **System Docs:** [`docs/CANONICAL_RULES_SYSTEM.md`](CANONICAL_RULES_SYSTEM.md)

---

## 💡 Key Principles to Remember

### 1. Physics-First (No Magic Numbers)

```python
# ❌ BAD: Static threshold
if energy > 0.8:
    enter_trade()

# ✅ GOOD: Rolling percentile
energy_pct = energy.rolling(window).rank(pct=True).iloc[-1]
if energy_pct > 0.8:  # Top 20% of recent distribution
    enter_trade()
```

### 2. Vectorization Over Loops

```python
# ❌ BAD: Python loop
for i in range(len(data)):
    result[i] = data[i] ** 2

# ✅ GOOD: NumPy vectorization
result = data ** 2
```

### 3. Data Safety (Never Lose User Data)

```python
# ❌ BAD: Direct write
df.to_csv("data/master/BTCUSD_H1.csv")

# ✅ GOOD: Atomic save with backup
from kinetra.persistence_manager import get_persistence_manager
pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)
pm.atomic_save(
    filepath="data/master/BTCUSD_H1.csv",
    content=df,
    writer=lambda path, data: data.to_csv(path, index=False)
)
```

### 4. Statistical Validation (p < 0.01)

```python
# ❌ BAD: Claims without proof
# "This strategy is better"

# ✅ GOOD: Validated claims
# "Strategy A outperforms B (p=0.003, n=100, effect_size=0.8)"
from scipy import stats
t_stat, p_value = stats.ttest_ind(strategy_a_results, strategy_b_results)
assert p_value < 0.01, f"Not statistically significant (p={p_value})"
```

---

## 🐛 Common Issues & Solutions

### Issue: Pre-commit hook fails

```bash
# See what failed
git commit -v

# Run linter manually
python scripts/lint_rules.py path/to/file.py

# Fix violations or add justification comment
# magic number ok - physically derived constant
```

### Issue: CI rules-validation fails

```bash
# Run linter locally to debug
python scripts/lint_rules.py kinetra/

# Check for specific violations
grep -rn "RSI\|MACD\|Bollinger" kinetra/
```

### Issue: Linter false positive

```python
# Add suppression comment
if value > 0.5:  # magic number ok - physical constant
    ...

# Or mark loop as unavoidable
for i in range(len(data)):  # vectorization unavoidable - stateful logic
    ...
```

---

## 🎓 Best Practices

### For Developers

1. **Read master rulebook** before starting new features
2. **Run linter frequently** during development
3. **Use quick references** for common patterns
4. **Question assumptions** - even "best practices"
5. **Validate claims** with statistical tests (p < 0.01)

### For Code Reviewers

1. **Check CI passes** rules-validation job
2. **Verify compliance** with canonical rules
3. **Question violations** if linter was bypassed
4. **Ensure new rules** go in master (not scattered)
5. **Validate tests** exist (100% coverage for new features)

### For AI Agents

1. **Always read master rulebook** at start of task
2. **Never regress** to conventional approaches
3. **No TA indicators** (RSI, MACD, etc.) - use physics
4. **No magic numbers** - use rolling percentiles
5. **When uncertain** - check master rulebook

---

## 📞 Questions?

### Where to Look

1. **Quick answer:** Check relevant quick reference file
2. **Complete answer:** Read [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)
3. **System details:** Read [`docs/CANONICAL_RULES_SYSTEM.md`](CANONICAL_RULES_SYSTEM.md)
4. **Still stuck:** Open GitHub discussion with `rules` tag

### Need Help?

- **Slack:** `#kinetra-dev` channel
- **GitHub:** Open issue with `rules` label
- **Docs:** Check `docs/` directory

---

## ✅ Checklist

Before you continue working:

- [ ] Read [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) (at least skim all sections)
- [ ] Installed pre-commit hook: `bash .github/SETUP_GIT.sh`
- [ ] Ran rules linter: `python scripts/lint_rules.py kinetra/`
- [ ] Understand hard prohibitions (no TA, no credentials, etc.)
- [ ] Bookmarked master rulebook for reference
- [ ] Know where to find quick references for your tools

---

## 🎉 Benefits

### For Developers

- ✅ **No confusion** - one source of truth
- ✅ **Automated checks** - catch violations early
- ✅ **Better code** - enforced best practices
- ✅ **Faster reviews** - consistent standards
- ✅ **Less debugging** - rules prevent common bugs

### For AI Agents

- ✅ **Clear guidelines** - no contradictions
- ✅ **Consistent behavior** - same rules everywhere
- ✅ **Better output** - follows project philosophy
- ✅ **Easier validation** - automated linting

### For the Project

- ✅ **Higher quality** - enforced standards
- ✅ **More maintainable** - consistent codebase
- ✅ **Faster onboarding** - clear documentation
- ✅ **Better collaboration** - shared understanding
- ✅ **Reduced bugs** - caught by linter/CI

---

## 🚀 Next Steps

1. **Complete checklist above**
2. **Start using the system** (it's already active!)
3. **Provide feedback** if you find issues
4. **Spread the word** to other team members

---

**Thank you for helping maintain Kinetra's high standards! 🎯**

**Questions? Check [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) or ask in #kinetra-dev**

---

*This announcement is part of the canonical rules consolidation initiative (January 2024)*