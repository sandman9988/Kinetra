# First Principles Research Framework

## Philosophy

**Assumption**: We know nothing about what drives markets.

**Approach**: Measure everything, validate empirically, let data reveal truth.

## Research Tools

### 1. Measurement Toolkit (`measurement_toolkit.py`)

**Purpose**: Validate that our measurements actually capture something real.

**What it does**:
- Extracts 100+ measurements from each instrument
- Tests which measurements predict future moves
- Compares measurement behavior across asset classes
- Identifies redundant vs useful features

**Run it**:
```bash
python scripts/research/measurement_toolkit.py
```

**Output**:
- `research_output/measurements/cross_class_comparison.csv`
- Correlation analysis between measurements and future returns
- Top predictive features per asset class

**Key Questions Answered**:
- Does "energy" actually predict anything?
- Do physics measurements outperform simple price/volume?
- Are forex dynamics different from crypto?

---

### 2. Fat Candle Forensics (`fat_candle_forensics.py`)

**Purpose**: Discover what triggers large market movements.

**What it does**:
- Identifies all "fat candles" (moves >3 ATR)
- Analyzes what happened 20 bars before each
- Compares preconditions to normal bars
- Tests if triggers are consistent across classes

**Run it**:
```bash
python scripts/research/fat_candle_forensics.py
```

**Output**:
- `research_output/fat_candles/forensics_results.json`
- Statistical tests (t-tests, effect sizes)
- Cross-class pattern analysis

**Key Questions Answered**:
- What precedes big crypto moves vs forex moves?
- Are the triggers the same across classes?
- Can we predict the next fat candle?

---

## Research Questions

### Q1: Do our physics measurements mean anything?

**Test**:
```bash
python scripts/research/measurement_toolkit.py
```

Look at the "Predictive Power" section. If "energy", "damping", "reynolds" don't correlate with future returns, they're useless.

**Hypothesis to validate**:
- High energy → future volatility?
- Low damping → continuation?
- High reynolds → trending?

---

### Q2: Are asset classes fundamentally different?

**Test**:
```bash
python scripts/research/fat_candle_forensics.py
```

Compare preconditions across classes:
- If crypto and forex share same triggers → universal principles
- If different triggers → need class-specific models

---

### Q3: What measurements actually matter?

**Test**:
```bash
python scripts/research/measurement_toolkit.py
```

Look at top 20 predictive features:
- Are they physics measurements?
- Or just price/volume derivatives?
- How many are redundant?

---

### Q4: Do marginal gains justify complexity?

**Methodology**:

1. Baseline: Price + Volume only
2. Level 2: + Velocity, Acceleration
3. Level 3: + Energy, Damping, Entropy
4. Level 4: + DSP, VPIN, Higher Moments

For each level:
- Train agent
- Measure Sharpe, drawdown, win rate
- Test on out-of-sample data

If Level 3 doesn't beat Level 2 by >10%, complexity not justified.

---

## Expected Insights

### What we might discover:

**Scenario A**: Physics measurements are predictive
- Keep them, refine them
- Build class-specific physics models
- Justify the complexity

**Scenario B**: Physics measurements are noise
- Drop them
- Focus on price/volume/time
- Simpler is better

**Scenario C**: Some physics useful, most aren't
- Feature ablation to keep top 5-10
- Drop the rest
- Reduce dimensionality

---

## Next Steps

### After running measurements:

1. **If physics is useful**:
   - Build class-specific physics models
   - Tune parameters per class/timeframe
   - Test marginal gains systematically

2. **If physics is useless**:
   - Pivot to pure price action
   - DSP/spectral analysis
   - Order flow proxies

3. **If mixed results**:
   - Feature selection (keep top N)
   - Ensemble models
   - Class-specific feature sets

---

## Research Workflow

```
Week 1: Run measurement_toolkit.py
└─> Identify predictive features
    └─> Drop useless measurements
        └─> Build reduced feature set

Week 2: Run fat_candle_forensics.py
└─> Identify move triggers per class
    └─> Test trigger consistency
        └─> Build class-specific models

Week 3: Feature ablation
└─> Train with increasing complexity
    └─> Measure marginal gains
        └─> Find optimal complexity

Week 4: Architecture comparison
└─> Single agent vs Triple agent
    └─> Class-agnostic vs Class-specific
        └─> Measure harvesting efficiency
```

---

## Output Files

### Measurement Toolkit:
- `research_output/measurements/cross_class_comparison.csv`
  - Rows: All instruments
  - Columns: Statistical properties per measurement
  - Use: Compare dynamics across classes

### Fat Candle Forensics:
- `research_output/fat_candles/forensics_results.json`
  - Per-class fat candle analysis
  - Precondition statistics
  - Significant measurements

---

## Validation Criteria

A measurement is **useful** if:
1. Correlation with future returns: |r| > 0.1
2. P-value < 0.05 (statistically significant)
3. Effect size (Cohen's d) > 0.3
4. Consistent across multiple instruments
5. Generalizes out-of-sample

A measurement is **useless** if:
1. No correlation with future moves
2. Only significant in one instrument
3. Doesn't generalize
4. Redundant with simpler measurement

---

## Philosophy Reminder

> "We think we understand markets, but we probably don't."

> "Measure everything. Trust nothing. Validate empirically."

> "If it doesn't predict, it doesn't matter."

---

## Contact

Questions? Check the research output files first.
Still confused? Review the philosophy section.
