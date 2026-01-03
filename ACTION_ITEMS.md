# 📋 Action Items - Canonical Rules System

**Date:** January 9, 2024  
**Priority:** High  
**Status:** Ready for Team Execution

---

## 🎯 Immediate Actions (Do Today)

### For All Team Members

- [ ] **Read the master rulebook** (15 minutes)
  ```bash
  cat AGENT_RULES_MASTER.md
  # Or open in your editor and skim all 17 sections
  ```

- [ ] **Install pre-commit hook** (30 seconds)
  ```bash
  bash .github/SETUP_GIT.sh
  ```

- [ ] **Run rules linter on your current work** (2 minutes)
  ```bash
  python scripts/lint_rules.py kinetra/
  # Fix any violations found
  ```

- [ ] **Bookmark key files** in your IDE
  - `AGENT_RULES_MASTER.md` (canonical source)
  - `.github/copilot-instructions.md` OR `.claude/instructions.md` (your quick ref)
  - `docs/CANONICAL_RULES_SYSTEM.md` (system guide)

---

## 🚀 First Week Tasks

### For Developers

- [ ] **Review existing PRs** against new rules
  - Run linter on PR branches
  - Fix any violations before merge
  - Update code to use `PersistenceManager.atomic_save()` if needed

- [ ] **Update your development workflow**
  - Add linter to your local testing routine
  - Use quick reference for common patterns
  - Check master rulebook when uncertain

- [ ] **Clean up any hardcoded patterns**
  ```bash
  # Check for magic numbers
  python scripts/lint_rules.py --warnings-as-errors
  
  # Replace with rolling percentiles or DSP-derived values
  ```

### For Code Reviewers

- [ ] **Add rules validation to review checklist**
  - Check CI `rules-validation` job passes
  - Verify no linter violations (or justified suppressions)
  - Ensure new features follow physics-first approach

- [ ] **Question rule violations**
  - If linter was bypassed with `--no-verify`
  - If suppression comments are unjustified
  - If code uses traditional TA indicators

### For AI Agent Users

- [ ] **Update AI agent context**
  - Reference `AGENT_RULES_MASTER.md` at start of sessions
  - Use appropriate quick reference file
  - Remind AI: "Follow rules in AGENT_RULES_MASTER.md"

- [ ] **Validate AI output**
  - Run linter on AI-generated code
  - Check for physics-first approach
  - Verify no magic numbers or TA indicators

---

## 📅 Ongoing Tasks

### Daily

- [ ] Run linter before committing
  ```bash
  python scripts/lint_rules.py <your_files>
  ```

- [ ] Check quick reference for common patterns

- [ ] Let pre-commit hook run (don't bypass)

### Weekly

- [ ] Review any linter warnings in your code
- [ ] Update suppressions if patterns change
- [ ] Check CI passes on all your branches

### Monthly

- [ ] Re-read relevant sections of master rulebook
- [ ] Provide feedback on rules system
- [ ] Suggest improvements or clarifications

---

## 🔧 Setup Validation Checklist

Verify your setup is complete:

```bash
# 1. Check pre-commit hook is installed
ls -la .git/hooks/pre-commit
# Should show symlink to .github/hooks/pre-commit

# 2. Verify linter works
python scripts/lint_rules.py --check-references
# Should output: ✅ No rule violations found!

# 3. Test on sample file
python scripts/lint_rules.py kinetra/physics_engine.py
# Should show any violations in that file

# 4. Check CI configuration
grep -A 5 "rules-validation:" .github/workflows/ci.yml
# Should show the rules-validation job

# 5. Verify master rulebook exists
cat AGENT_RULES_MASTER.md | head -20
# Should show the canonical rulebook header
```

All commands should work without errors.

---

## 📢 Communication Tasks

### For Team Leads

- [ ] **Announce to team** (use `docs/RULES_CONSOLIDATION_ANNOUNCEMENT.md`)
  - Slack/Teams message
  - Email summary
  - Team meeting agenda item

- [ ] **Schedule Q&A session** (optional)
  - Review master rulebook structure
  - Demo linter usage
  - Answer questions

- [ ] **Update onboarding docs**
  - Add "Read AGENT_RULES_MASTER.md" to checklist
  - Add "Install pre-commit hook" to setup steps
  - Link to `docs/CANONICAL_RULES_SYSTEM.md`

### For All Team Members

- [ ] **Ask questions** if anything is unclear
  - GitHub discussions with `rules` tag
  - Slack `#kinetra-dev` channel
  - Direct message to team leads

- [ ] **Report issues** with linter or rules
  - False positives
  - Missing patterns
  - Unclear documentation

---

## 🐛 Known Issues to Address

### Existing Code Violations

Run full project scan and track issues:

```bash
# Generate full report
python scripts/lint_rules.py kinetra/ > rules_violations.txt

# Prioritize by severity
grep "ERROR:" rules_violations.txt | wc -l  # Count errors
grep "WARNING:" rules_violations.txt | wc -l  # Count warnings
```

Action items:
- [ ] Review all ERROR-level violations
- [ ] Create issues for legitimate violations
- [ ] Add suppression comments for justified cases
- [ ] Fix critical violations (hardcoded credentials, TA indicators)

### Documentation Gaps

- [ ] Add examples for common patterns to master rulebook
- [ ] Create FAQ section based on team questions
- [ ] Add more code snippets to quick references
- [ ] Document linter suppression best practices

---

## ✅ Success Metrics

Track these over the next 30 days:

### Adoption Metrics

- [ ] % of team members who have read master rulebook
- [ ] % of commits that pass pre-commit hook
- [ ] % of PRs that pass rules-validation CI job

### Quality Metrics

- [ ] # of rule violations found by linter (should decrease)
- [ ] # of hardcoded credentials found (should be 0)
- [ ] # of traditional TA indicators added (should be 0)

### Feedback Metrics

- [ ] # of false positives reported (aim to fix)
- [ ] # of unclear rules (aim to clarify)
- [ ] Team satisfaction with system (survey)

---

## 🎓 Training & Support

### Resources Available

- **Master Rulebook:** `AGENT_RULES_MASTER.md` (comprehensive)
- **System Guide:** `docs/CANONICAL_RULES_SYSTEM.md` (how it works)
- **Team Announcement:** `docs/RULES_CONSOLIDATION_ANNOUNCEMENT.md` (what changed)
- **Quick References:** `.github/copilot-instructions.md`, `.claude/instructions.md`

### Getting Help

1. **Quick questions:** Check quick reference
2. **Detailed questions:** Check master rulebook
3. **System questions:** Check system guide
4. **Still stuck:** Ask in Slack or GitHub discussions

### Office Hours (Optional)

- [ ] Schedule weekly "rules office hours"
- [ ] Team lead available for questions
- [ ] Demo sessions for complex patterns

---

## 🚫 Common Pitfalls to Avoid

### Don't Do This

- ❌ **Bypass pre-commit hook** without good reason
  ```bash
  git commit --no-verify  # Only in emergencies!
  ```

- ❌ **Add TA indicators** (RSI, MACD, BB, etc.)
  ```python
  # This will fail CI
  df['rsi'] = ta.RSI(df['close'], timeperiod=14)
  ```

- ❌ **Hardcode credentials** in code
  ```python
  # This will fail CI
  API_KEY = "sk-1234567890abcdef"
  ```

- ❌ **Use magic numbers** without justification
  ```python
  # This will warn
  if energy > 0.8:  # Static threshold - not adaptive!
  ```

- ❌ **Edit quick references** instead of master
  ```bash
  # Wrong: Edit .github/copilot-instructions.md
  # Right: Edit AGENT_RULES_MASTER.md, then update quick refs
  ```

### Do This Instead

- ✅ **Fix violations** before committing
- ✅ **Use physics-based features** (energy, Reynolds, entropy)
- ✅ **Use environment variables** for credentials
- ✅ **Use rolling percentiles** for thresholds
- ✅ **Edit master rulebook** for rule changes

---

## 📊 30-Day Review Checklist

Schedule review on **February 8, 2024**:

- [ ] **Adoption:** How many team members using system?
- [ ] **Effectiveness:** Rule violations trending down?
- [ ] **Feedback:** What issues have been reported?
- [ ] **Improvements:** What needs to be added/changed?
- [ ] **Documentation:** Is anything unclear?
- [ ] **Automation:** Are checks catching issues?

Action from review:
- [ ] Update master rulebook based on feedback
- [ ] Enhance linter patterns if needed
- [ ] Add examples for common questions
- [ ] Celebrate wins! 🎉

---

## 🎯 Quick Start Checklist (New Team Members)

If you're new to the project:

1. [ ] Read `AGENT_RULES_MASTER.md` (at least skim all sections)
2. [ ] Install pre-commit hook: `bash .github/SETUP_GIT.sh`
3. [ ] Read your tool's quick reference (Copilot or Claude)
4. [ ] Run linter on sample code: `python scripts/lint_rules.py kinetra/physics_engine.py`
5. [ ] Understand hard prohibitions (no TA, no credentials, etc.)
6. [ ] Bookmark master rulebook for reference
7. [ ] Ask questions in Slack if anything is unclear

**Estimated time:** 30 minutes

---

## 📞 Contact & Support

### Questions About:

- **Rules content:** Check master rulebook, then ask team lead
- **Linter issues:** Create GitHub issue with `linter` label
- **System design:** Read `docs/CANONICAL_RULES_SYSTEM.md`
- **Setup problems:** Ask in Slack `#kinetra-dev`

### Feedback & Suggestions

We want your input! Please share:
- What's working well
- What's confusing
- What's missing
- Ideas for improvement

**Channels:**
- GitHub discussions (tag: `rules`)
- Slack `#kinetra-dev`
- Direct message to team leads

---

## ✨ Thank You!

Your cooperation in adopting this system will help maintain Kinetra's high standards and ensure we're all working from the same playbook. 

**Let's build something great together! 🚀**

---

**Status:** Active  
**Last Updated:** January 9, 2024  
**Next Review:** February 8, 2024

---

*For complete information, see `AGENT_RULES_MASTER.md` or `docs/CANONICAL_RULES_SYSTEM.md`*