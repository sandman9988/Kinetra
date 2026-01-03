# Canonical Rules System - Developer Guide

**Version:** 2.0  
**Last Updated:** 2024-01-09  
**Status:** Active

---

## Overview

The Kinetra project uses a **single canonical rulebook** to consolidate all agent rules, coding standards, and development guidelines. This eliminates confusion from scattered documentation and ensures all AI agents and developers follow the same authoritative source.

---

## 📚 The Canonical Rulebook

### Location
**[`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)** (project root)

This is the **single source of truth** for all:
- Core philosophy (physics-first, no magic numbers)
- Data safety & integrity rules
- Performance & optimization guidelines
- Security & hard prohibitions
- Testing requirements
- Code quality standards
- All domain-specific rules (MetaAPI, backtesting, etc.)

### Quick References

Other instruction files provide **quick references** and point to the master:

| File | Purpose | Audience |
|------|---------|----------|
| [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) | **Canonical source** | All developers & AI agents |
| [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) | Quick reference | GitHub Copilot |
| [`.claude/instructions.md`](../.claude/instructions.md) | Quick reference | Claude/Zed AI |
| [`.claude/type_checking_guidelines.md`](../.claude/type_checking_guidelines.md) | Type checking specifics | BasedPyRight users |

---

## 🎯 Why a Canonical Rulebook?

### Problems Solved

**Before:** 
- Rules scattered across 8+ files
- Contradictions between sources
- AI agents got confused about which rules to follow
- Developers had to search multiple locations
- Outdated copies of rules in archive folders

**After:**
- ✅ Single source of truth
- ✅ No contradictions (all consolidated)
- ✅ Clear hierarchy (master → quick references)
- ✅ Easy to maintain (edit one file)
- ✅ Automated enforcement (CI validates references)

---

## 🔧 How It Works

### 1. Rule Hierarchy

```
┌─────────────────────────────────────┐
│   AGENT_RULES_MASTER.md (ROOT)      │
│   ═══════════════════════════════   │
│   Single Source of Truth            │
│   All 17 rule sections              │
│   Comprehensive & authoritative     │
└──────────────┬──────────────────────┘
               │
               ├─────────────────┬─────────────────┬──────────────────┐
               ▼                 ▼                 ▼                  ▼
       ┌───────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐
       │ Copilot Quick │  │ Claude     │  │ Type Checking│  │ Archive      │
       │ Reference     │  │ Quick Ref  │  │ Guidelines   │  │ (Deprecated) │
       └───────────────┘  └────────────┘  └──────────────┘  └──────────────┘
          (Subset)           (Subset)         (Subset)        (References master)
```

### 2. Content Strategy

**Master Rulebook Contains:**
- Complete, comprehensive rules (100%)
- All 17 major sections
- Full details, examples, justifications
- ~1500+ lines of canonical truth

**Quick References Contain:**
- 10-20% of master content
- Most critical rules for specific tool
- **Link to master for complete rules**
- Tool-specific formatting/examples

### 3. Enforcement Mechanisms

#### A. Automated CI Validation

Every commit triggers checks (`rules-validation` job):

```yaml
# .github/workflows/ci.yml
- name: Check canonical rules references
  run: python scripts/lint_rules.py --check-references

- name: Lint code against rules
  run: python scripts/lint_rules.py kinetra/

- name: Check for banned patterns
  run: grep -rn "RSI|MACD|Bollinger" kinetra/
```

#### B. Pre-Commit Hook

Before every commit:

```bash
# .github/hooks/pre-commit
1. Verify canonical rulebook exists
2. Run rules linter on staged files
3. Check for hardcoded credentials
4. Check for traditional TA indicators
5. Backup data if needed
6. Validate all checks pass
```

#### C. Rules Linter (`scripts/lint_rules.py`)

Validates code against canonical rules:

```bash
# Lint entire project
python scripts/lint_rules.py

# Lint specific files
python scripts/lint_rules.py kinetra/physics_engine.py

# Check only references
python scripts/lint_rules.py --check-references
```

**Checks for:**
- ❌ Traditional TA indicators (RSI, MACD, BB, etc.)
- ❌ Magic numbers (static thresholds)
- ❌ Hardcoded credentials
- ❌ Unsafe data operations (direct file writes)
- ❌ Unseeded RNG in backtests
- ⚠️  Python loops that should be vectorized

---

## 📋 The 17 Rule Sections

The canonical rulebook is organized into 17 major sections:

1. **Meta-Rules** - Conversation continuity & compression
2. **Core Philosophy** - First principles, physics-first
3. **Data Safety & Integrity** - #1 priority (never lose user data)
4. **Performance - Vectorization** - NumPy/Pandas over loops
5. **Memory & Efficiency** - Lazy evaluation, memory-mapped files
6. **I/O & Concurrency** - Async I/O, batching
7. **Determinism & Reproducibility** - RNG seeding, locked dependencies
8. **Backtesting Engine** - Monte Carlo, statistical significance
9. **Experiment Safety** - Validation gates, circuit breakers
10. **MetaAPI Connector** - Historical data, paper trading
11. **Logging & Error Handling** - Structured logs, error recovery
12. **Security & Hard Prohibitions** - No credentials, no live orders
13. **Code Quality & Style** - PEP 8, Black, Ruff, type hints
14. **Type Checking** - BasedPyRight, Optional handling
15. **Physics-First Approach** - Energy, entropy, Reynolds, etc.
16. **Testing Requirements** - 100% coverage, p < 0.01
17. **Deliverables & Validation** - Performance targets

---

## 🚀 Quick Start for Developers

### First Time Setup

1. **Read the master rulebook:**
   ```bash
   cat AGENT_RULES_MASTER.md
   ```

2. **Install pre-commit hook:**
   ```bash
   bash .github/SETUP_GIT.sh
   ```

3. **Run rules linter:**
   ```bash
   python scripts/lint_rules.py
   ```

### Daily Workflow

1. **Before coding:** Check relevant quick reference for your tool
2. **During coding:** Follow rules (linter will catch violations)
3. **Before committing:** Pre-commit hook validates automatically
4. **If confused:** Check master rulebook for complete details

### Adding New Rules

1. **Edit master rulebook only:**
   ```bash
   vim AGENT_RULES_MASTER.md
   ```

2. **Update quick references if needed:**
   - Keep them minimal (10-20% of master)
   - Always link back to master

3. **Validate changes:**
   ```bash
   python scripts/lint_rules.py --check-references
   ```

4. **Commit with explanation:**
   ```bash
   git commit -m "Add rule: <description> to AGENT_RULES_MASTER.md"
   ```

---

## 🔍 Rules Linter Details

### What It Checks

#### ERROR-level violations (CI fails):
- Traditional TA indicators (RSI, MACD, Bollinger, ADX, etc.)
- Hardcoded credentials (API keys, secrets, passwords)
- Unsafe data operations (direct CSV writes without atomic save)
- Unseeded RNG in backtest code
- Missing canonical rulebook references

#### WARNING-level violations (informational):
- Magic numbers (static thresholds)
- Python loops that could be vectorized
- Fixed rolling windows (should be DSP-derived)

### Usage Examples

```bash
# Lint entire project
python scripts/lint_rules.py

# Lint specific directory
python scripts/lint_rules.py kinetra/rl/

# Lint specific file
python scripts/lint_rules.py kinetra/physics_engine.py

# Only check canonical references
python scripts/lint_rules.py --check-references

# Treat warnings as errors
python scripts/lint_rules.py --warnings-as-errors
```

### Suppressing Violations

If a rule violation is unavoidable (rare), add a comment:

```python
# Static threshold OK - physically derived constant
if energy > 0.5:  # magic number ok
    ...

# Vectorization unavoidable - complex stateful logic
for i in range(len(data)):  # loop required
    ...
```

---

## 🛡️ Hard Prohibitions

These rules are **NEVER negotiable** (CI will fail):

| Prohibition | Rationale | Enforcement |
|-------------|-----------|-------------|
| **NO TA indicators** | Not physics-based | Rules linter + CI grep |
| **NO hardcoded credentials** | Security violation | Rules linter + CI grep |
| **NO live order placement** | Backtest/paper only | Rules linter |
| **NO data loss** | Must use atomic saves | Rules linter |
| **NO unseeded RNG in backtests** | Must be deterministic | Rules linter |

---

## 📊 CI Validation Jobs

The CI pipeline includes a dedicated `rules-validation` job:

```yaml
rules-validation:
  name: Validate Against Canonical Rules
  steps:
    - Check canonical rules references
    - Lint code against rules
    - Check for banned patterns (TA indicators)
    - Check for hardcoded credentials
    - Check for unsafe data operations
    - Verify AGENT_RULES_MASTER.md exists
```

**All checks must pass** before merge to `main` or `develop`.

---

## 🔄 Migration from Old System

If you find old rule files in the codebase:

### Deprecated Files (Reference Master Only)

- `archive/status-reports/AI_AGENT_INSTRUCTIONS.md` ⚠️ Deprecated
- Any `*_RULES.md` in archive folders ⚠️ Deprecated
- Standalone rule fragments ⚠️ Check if in master

### Action Required

1. **Check if rule is in master:**
   ```bash
   grep -i "your_rule" AGENT_RULES_MASTER.md
   ```

2. **If missing, add to master:**
   - Don't create new rule files
   - Add to appropriate section in master

3. **Update deprecated files:**
   - Add deprecation notice
   - Link to master rulebook

---

## 🎓 Best Practices

### For AI Agents

1. **Always read master rulebook** at start of task
2. **Never regress** to conventional approaches (e.g., adding RSI)
3. **Question assumptions** - even "best practices"
4. **Validate everything** - p < 0.01 for statistical claims
5. **When in doubt** - check master rulebook section

### For Human Developers

1. **Familiarize with master rulebook** structure
2. **Use quick references** for daily work
3. **Bookmark master** for deep dives
4. **Run linter frequently** during development
5. **Update master** when adding rules (not quick refs)

### For Code Reviewers

1. **Verify master compliance** in PRs
2. **Check CI passes** rules-validation job
3. **Question rule violations** if linter bypassed
4. **Ensure new rules** go in master (not scattered)
5. **Validate test coverage** (100% for new features)

---

## 🐛 Troubleshooting

### Pre-commit hook fails

```bash
# Check what failed
git commit -v

# Run linter manually
python scripts/lint_rules.py <your_files>

# Skip hook (NOT recommended)
git commit --no-verify
```

### CI rules-validation job fails

1. **Check CI logs** for specific violation
2. **Run linter locally:**
   ```bash
   python scripts/lint_rules.py kinetra/
   ```
3. **Fix violations** or add suppression comments
4. **Re-run CI** by pushing again

### Linter false positive

1. **Add suppression comment:**
   ```python
   # magic number ok - physically derived
   ```
2. **If systematic issue:** Update linter patterns in `scripts/lint_rules.py`
3. **Report issue:** Open GitHub issue with example

---

## 📈 Future Enhancements

### Planned

- [ ] **Auto-sync quick refs** from master (extraction tool)
- [ ] **Rule deprecation workflow** (mark old rules, migration guide)
- [ ] **Conflict detection** (check for contradictions)
- [ ] **Rules coverage report** (what % of code follows each rule)
- [ ] **Interactive rules browser** (searchable web UI)

### Proposed

- [ ] **Rule impact analysis** (when changing a rule, show affected code)
- [ ] **Rule versioning** (track changes over time)
- [ ] **Custom rule plugins** (project-specific extensions)
- [ ] **Rules playground** (test code snippets against rules)

---

## 📞 Getting Help

### Questions?

1. **Read master rulebook:** Most answers are there
2. **Search issues:** `is:issue label:rules`
3. **Ask in discussions:** Tag with `rules` label
4. **Check this guide:** You're reading it!

### Found a Bug?

1. **Linter bug:** Open issue with example code
2. **Rule contradiction:** Open issue with conflicting rules
3. **Missing rule:** Propose addition to master

### Contributing

See [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) for contribution guidelines.

---

## 📜 License

This documentation is part of the Kinetra project and follows the same license.

---

**Remember:** When in doubt, check [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) - it's the ultimate authority!