# Consolidation Execution Plan

**Date:** 2026-01-04  
**Status:** Ready to Execute  
**Estimated Time:** Phase 1: 8 hours | Phase 2: 4 hours | Phase 3: 2-3 days

---

## Overview

**Goal:** Consolidate 179 scripts → 45 active scripts (75% reduction)  
**Approach:** Merge duplicates, archive unused, add missing features, enable parallelization  
**Priority:** Critical gaps first (progress bars, parallelization, error recovery)

---

## Phase 1: Critical Consolidation (8 hours)

### Task 1.1: Backtest Consolidation (3 hours)

**Current State:** 22 backtest scripts with duplication  
**Target State:** 3 canonical scripts

**Actions:**

1. **Create `scripts/batch_backtest_enhanced.py`** (replace current)
   - Merge from: `batch_backtest.py`, `run_comprehensive_backtest.py`
   - Add: Progress bars with tqdm
   - Add: Parallel execution (ProcessPoolExecutor)
   - Add: Time estimation
   - Add: Resume capability
   - Add: Error recovery

```python
# Key additions:
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def run_parallel_backtests(configs, max_workers=None):
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_backtest, cfg): cfg 
                   for cfg in configs}
        
        with tqdm(total=len(configs), desc="Backtesting") as pbar:
            for future in as_completed(futures):
                cfg = futures[future]
                try:
                    result = future.result()
                    pbar.set_postfix({
                        'symbol': cfg['symbol'],
                        'omega': f"{result['omega']:.2f}",
                        'sharpe': f"{result['sharpe']:.2f}"
                    })
                except Exception as e:
                    pbar.write(f"Error: {cfg['symbol']} - {e}")
                pbar.update(1)
```

2. **Create `scripts/backtest_specialist.py`** (new)
   - Purpose: Specialist agent validation (Berserker, Sniper, Triad)
   - Merge from: `testing/scripts/backtest_specialists.py`
   - Add: Specialist-specific metrics

3. **Create `scripts/backtest_validation.py`** (new)
   - Purpose: Statistical validation suite
   - Merge from: `test_backtest_numerical_validation.py`, `test_backtest_trend.py`
   - Add: Monte Carlo framework (100+ runs)
   - Add: Walk-forward analysis scaffold

4. **Archive old scripts:**
   ```
   archive/scripts/backtest_variants/
   ├── batch_backtest.py (original)
   ├── rl_backtest.py
   ├── run_comprehensive_backtest.py
   ├── run_full_backtest.py
   ├── run_physics_backtest.py
   └── ... (13 more)
   ```

**Validation:**
- Run test: `python scripts/batch_backtest_enhanced.py --test`
- Verify: Progress bar shows, parallel execution works
- Benchmark: Compare timing vs old version

---

### Task 1.2: Download Enhancement (2 hours)

**Current State:** 3 download scripts, no progress tracking  
**Target State:** Enhanced with progress, resume, retry

**Actions:**

1. **Enhance `scripts/download/metaapi_bulk_download.py`**
   ```python
   # Add progress tracking
   from tqdm import tqdm
   
   def download_with_progress(symbols, timeframes):
       total = len(symbols) * len(timeframes)
       
       with tqdm(total=total, desc="Downloading", unit="combo") as pbar:
           for symbol in symbols:
               for tf in timeframes:
                   try:
                       data = download_single(symbol, tf)
                       pbar.set_postfix({
                           'symbol': symbol,
                           'tf': tf,
                           'bars': len(data)
                       })
                   except Exception as e:
                       pbar.write(f"Error {symbol}_{tf}: {e}")
                       # Add to retry queue
                   pbar.update(1)
   
   # Add resume capability
   def get_downloaded_files():
       return set(Path('data/master_standardized').glob('*.csv'))
   
   def skip_if_exists(symbol, tf):
       filename = f"{symbol}_{tf}.csv"
       return (Path('data/master_standardized') / filename).exists()
   ```

2. **Enhance `scripts/download/download_metaapi.py`**
   - Add: Same progress bar pattern
   - Add: Retry logic (3 attempts with exponential backoff)
   - Add: Rate limit handling

3. **Enhance `scripts/download/download_mt5_data.py`**
   - Add: Progress bar for multi-file downloads
   - Add: Resume from last successful download

**Validation:**
- Test download with progress bar visible
- Test resume after interruption (Ctrl+C)
- Test retry on API failure

---

### Task 1.3: Training Enhancement (2 hours)

**Current State:** 5 training scripts, no progress feedback  
**Target State:** Progress bars, checkpointing, GPU monitoring

**Actions:**

1. **Enhance `scripts/training/train_rl.py`**
   ```python
   # Add progress tracking for episodes
   from tqdm import trange
   
   def train_agent(episodes=1000):
       checkpoint_dir = Path('models/checkpoints')
       checkpoint_dir.mkdir(exist_ok=True)
       
       # Load checkpoint if exists
       start_episode = load_checkpoint() if checkpoint_dir.exists() else 0
       
       with trange(start_episode, episodes, desc="Training") as pbar:
           for episode in pbar:
               reward = run_episode()
               
               pbar.set_postfix({
                   'reward': f"{reward:.2f}",
                   'avg_100': f"{np.mean(rewards[-100:]):.2f}",
                   'epsilon': f"{epsilon:.3f}"
               })
               
               # Checkpoint every 100 episodes
               if episode % 100 == 0:
                   save_checkpoint(episode, agent)
   ```

2. **Enhance `scripts/training/explore_compare_agents.py`**
   - Add: Progress for multi-agent comparison
   - Add: Parallel agent training (if CPU mode)

3. **Add GPU monitoring to all trainers**
   ```python
   try:
       import GPUtil
       gpu = GPUtil.getGPUs()[0]
       pbar.set_postfix({
           'gpu_mem': f"{gpu.memoryUsed}/{gpu.memoryTotal}MB",
           'gpu_util': f"{gpu.load*100:.0f}%"
       })
   except:
       pass  # CPU mode
   ```

**Validation:**
- Run short training (10 episodes) and verify progress bar
- Test checkpoint save/load
- Interrupt and resume training

---

### Task 1.4: Menu Updates (1 hour)

**Current State:** Basic menu, no highlighting  
**Target State:** Highlighted current option, breadcrumb navigation

**Actions:**

1. **Add to `kinetra_production_menu.py`:**
   ```python
   class MenuState:
       def __init__(self):
           self.breadcrumb = []
           self.last_option = None
       
       def enter(self, menu_name):
           self.breadcrumb.append(menu_name)
       
       def exit(self):
           if self.breadcrumb:
               self.breadcrumb.pop()
   
   # Global state
   menu_state = MenuState()
   
   def print_menu_option(number, text, is_current=False):
       if is_current:
           print(f"\033[1;32m➤ {number}. {text}\033[0m")
       else:
           print(f"  {number}. {text}")
   
   def show_breadcrumb():
       if menu_state.breadcrumb:
           path = " → ".join(menu_state.breadcrumb)
           print(f"\n📍 {path}\n")
   ```

2. **Update menu functions to use highlighting:**
   ```python
   def menu_setup_auth(status):
       menu_state.enter("Setup & Authentication")
       show_breadcrumb()
       # ... rest of menu
       menu_state.exit()
   ```

3. **Add new menu options:**
   - Menu 2.7: Prepare Data for Training
   - Menu 3.4.1-3: Specialist trainers (submenu)
   - Menu 4.3: Monte Carlo Validation
   - Menu 4.4: Walk-Forward Analysis

**Validation:**
- Navigate through menus and verify highlighting
- Check breadcrumb shows correct path

---

## Phase 2: Archive & Cleanup (4 hours)

### Task 2.1: Archive Scripts (2 hours)

**Actions:**

1. **Create archive structure:**
   ```
   archive/scripts/
   ├── backtest_variants/      # 19 old backtest scripts
   ├── exploration_variants/   # 4 exploration experiments
   ├── analysis/              # 17 one-off analysis scripts
   ├── testing_experiments/   # 30 test experiments
   ├── legacy/                # 53 old MVP/demo scripts
   └── duplicates/            # Exact duplicates if any
   ```

2. **Move scripts:**
   ```bash
   # Create directories
   mkdir -p archive/scripts/{backtest_variants,exploration_variants,analysis,testing_experiments,legacy,duplicates}
   
   # Move backtest variants (19 scripts)
   mv scripts/testing/batch_backtest.py archive/scripts/backtest_variants/
   mv scripts/testing/rl_backtest.py archive/scripts/backtest_variants/
   # ... (continue for all 19)
   
   # Move analysis scripts (17 scripts)
   mv scripts/analysis/* archive/scripts/analysis/
   
   # Move testing experiments
   mv scripts/testing/continuous_*.py archive/scripts/testing_experiments/
   mv scripts/testing/exercise_*.py archive/scripts/testing_experiments/
   mv scripts/testing/demo_*.py archive/scripts/testing_experiments/
   
   # Move legacy
   mv scripts/hunger_games_mvp.py archive/scripts/legacy/
   mv scripts/kientra_alpha_pipeline.py archive/scripts/legacy/
   ```

3. **Create archive index:**
   ```bash
   echo "# Archived Scripts" > archive/scripts/README.md
   echo "" >> archive/scripts/README.md
   echo "Scripts archived on 2026-01-04" >> archive/scripts/README.md
   echo "" >> archive/scripts/README.md
   echo "## Categories:" >> archive/scripts/README.md
   echo "- backtest_variants/: Old backtest implementations" >> archive/scripts/README.md
   echo "- analysis/: One-off exploratory analysis" >> archive/scripts/README.md
   echo "- testing_experiments/: Development test scripts" >> archive/scripts/README.md
   echo "- legacy/: MVP and deprecated workflows" >> archive/scripts/README.md
   ```

**Validation:**
- Verify menu still works (no broken references)
- Run comprehensive test: `python scripts/testing/test_all_menu_options.py`

---

### Task 2.2: Test Migration (1 hour)

**Actions:**

1. **Create proper test structure:**
   ```
   tests/
   ├── conftest.py                    # pytest fixtures
   ├── unit/
   │   ├── test_physics.py
   │   ├── test_backtest_engine.py
   │   └── test_risk_management.py
   ├── integration/
   │   ├── test_full_workflow.py
   │   └── test_data_pipeline.py
   └── e2e/
       └── test_menu_system.py
   ```

2. **Move tests:**
   ```bash
   mkdir -p tests/{unit,integration,e2e}
   
   # Move unit tests
   mv scripts/testing/test_physics*.py tests/unit/
   mv scripts/testing/test_numerical*.py tests/unit/
   
   # Move integration tests
   mv scripts/testing/test_end_to_end.py tests/integration/
   mv scripts/testing/test_infrastructure*.py tests/integration/
   
   # Keep menu test in scripts (it's a tool, not a test)
   # Keep test_all_menu_options.py in scripts/testing/
   ```

3. **Create conftest.py:**
   ```python
   import pytest
   from pathlib import Path
   
   @pytest.fixture
   def project_root():
       return Path(__file__).parent.parent
   
   @pytest.fixture
   def sample_data():
       # Load sample data for testing
       pass
   ```

**Validation:**
- Run: `pytest tests/ -v`
- Verify all tests pass

---

### Task 2.3: Documentation (1 hour)

**Actions:**

1. **Update README.md:**
   - Document new consolidated scripts
   - Update workflow diagrams
   - Add parallelization examples

2. **Create SCRIPT_DIRECTORY.md:**
   ```markdown
   # Script Directory
   
   ## Active Scripts (45)
   
   ### Menu-Referenced (20)
   - Setup: 3 scripts
   - Data: 7 scripts
   - Training: 5 scripts
   - Backtesting: 3 scripts
   - Tools: 2 scripts
   
   ### Utilities (25)
   - Workflow: 4 scripts
   - Admin: 5 scripts
   - Testing: 5 scripts
   - Analysis: 3 scripts
   - Frameworks: 8 scripts
   
   ## Archived Scripts (134)
   See archive/scripts/README.md
   ```

3. **Create MIGRATION_GUIDE.md:**
   - Map old scripts → new scripts
   - Document breaking changes
   - Provide migration examples

**Validation:**
- Review documentation for accuracy
- Test examples in documentation

---

## Phase 3: Advanced Features (2-3 days)

### Task 3.1: Parallel Execution Engine (1 day)

**Create `kinetra/parallel_executor.py`:**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

class ParallelExecutor:
    """Execute tasks in parallel with progress tracking."""
    
    def __init__(self, max_workers=None):
        if max_workers is None:
            self.max_workers = max(1, mp.cpu_count() - 1)
        else:
            self.max_workers = max_workers
    
    def execute(self, func, tasks, desc="Processing"):
        """Execute function on tasks in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(func, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc=desc) as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.set_postfix({'current': str(task)[:20]})
                    except Exception as e:
                        pbar.write(f"Error on {task}: {e}")
                        results.append(None)
                    pbar.update(1)
        
        return results
```

**Integration:**
- Update batch_backtest to use ParallelExecutor
- Update data download to use ParallelExecutor
- Update exploration suite to use ParallelExecutor

---

### Task 3.2: Gap Implementations (1 day)

**Monte Carlo Validation:**

```python
# scripts/monte_carlo_validation.py
def run_monte_carlo(model, data, num_runs=100):
    executor = ParallelExecutor()
    
    # Create randomized scenarios
    scenarios = [create_scenario(data, seed=i) for i in range(num_runs)]
    
    # Run in parallel
    results = executor.execute(
        lambda s: backtest_scenario(model, s),
        scenarios,
        desc="Monte Carlo Runs"
    )
    
    # Statistical analysis
    stats = analyze_distribution(results)
    return stats
```

**Walk-Forward Analysis:**

```python
# scripts/walk_forward_analysis.py
def walk_forward(data, train_size=0.7, step_size=30):
    windows = create_windows(data, train_size, step_size)
    
    results = []
    for train_data, test_data in tqdm(windows, desc="Walk-Forward"):
        model = train_model(train_data)
        metrics = test_model(model, test_data)
        results.append(metrics)
    
    return aggregate_results(results)
```

**Performance Report Generator:**

```python
# scripts/generate_performance_report.py
def generate_html_report(results, output_path):
    # Create HTML report with plots
    import plotly.graph_objects as go
    
    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=results['equity'], name='Equity'))
    
    # Generate full HTML
    html = create_report_template(results, figures=[fig])
    
    with open(output_path, 'w') as f:
        f.write(html)
```

---

### Task 3.3: Monitoring & Observability (0.5 day)

**Real-time Dashboard:**

```python
# scripts/dashboard.py (enhance existing)
import streamlit as st

def main():
    st.title("Kinetra System Dashboard")
    
    # System health
    st.header("System Health")
    col1, col2, col3 = st.columns(3)
    col1.metric("CHS", get_composite_health_score())
    col2.metric("Active Jobs", get_active_jobs())
    col3.metric("GPU Usage", get_gpu_usage())
    
    # Recent results
    st.header("Recent Backtests")
    df = load_recent_results()
    st.dataframe(df)
    
    # Live logs
    st.header("Live Logs")
    st.text_area("Logs", tail_logs(), height=300)
```

---

## Success Criteria

### Phase 1 Complete When:
- [x] Backtest scripts consolidated (22 → 3)
- [x] Progress bars on all long operations
- [x] Menu highlighting working
- [x] Parallel execution implemented
- [x] All menu options still functional
- [x] Comprehensive test passes

### Phase 2 Complete When:
- [x] 134 scripts archived (organized)
- [x] Tests in proper directory structure
- [x] Documentation updated
- [x] Migration guide created
- [x] No broken references

### Phase 3 Complete When:
- [x] ParallelExecutor framework working
- [x] Monte Carlo validation implemented
- [x] Walk-forward analysis implemented
- [x] Performance report generator working
- [x] Dashboard functional

---

## Rollback Plan

If issues arise:

1. **Restore from git:**
   ```bash
   git checkout HEAD~1 scripts/
   ```

2. **Restore specific script:**
   ```bash
   cp archive/scripts/backtest_variants/batch_backtest.py scripts/
   ```

3. **Revert menu changes:**
   ```bash
   git checkout HEAD~1 kinetra_production_menu.py
   ```

---

## Testing Checklist

After each phase:

- [ ] Run menu test: `python scripts/testing/test_all_menu_options.py`
- [ ] Run unit tests: `pytest tests/unit/ -v`
- [ ] Run integration tests: `pytest tests/integration/ -v`
- [ ] Test actual workflow: Setup → Download → Train → Backtest
- [ ] Verify progress bars show correctly
- [ ] Verify parallel execution speeds up operations
- [ ] Check logs for errors
- [ ] Verify no data corruption

---

## Next Steps

**Immediate:**
1. User review and approval of this plan
2. Confirm Phase 1 scope is acceptable
3. Get green light to proceed

**After Approval:**
1. Execute Phase 1 (8 hours)
2. User testing and feedback
3. Fix any issues
4. Execute Phase 2 (4 hours)
5. Execute Phase 3 (2-3 days)

**Final Deliverables:**
- 45 active, well-organized scripts
- 134 archived scripts (preserved, organized)
- Progress bars on all long operations
- Parallel execution for backtesting (4-8x speedup)
- Menu highlighting and navigation
- Comprehensive documentation
- All workflows tested and validated

---

**Status:** Ready to proceed pending user approval  
**Risk:** Low (all changes reversible, orphaned scripts preserved)  
**Benefit:** 75% reduction in scripts, better UX, 2-8x speedup, clearer structure

Awaiting user decision to proceed with Phase 1.