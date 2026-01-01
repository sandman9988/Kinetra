#!/usr/bin/env python3
"""
Menu System Audit Tool
======================

Comprehensive audit that:
1. Maps all menu options to their implementation
2. Identifies scripts called vs. scripts available
3. Finds deadweight (unused scripts)
4. Finds unimplemented menu options
5. Generates visual mapping

Usage:
    python scripts/audit_menu_system.py
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# MENU PARSER
# =============================================================================

class MenuAuditor:
    """Audit menu system and map to scripts."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.menu_file = project_root / "kinetra_menu.py"
        self.scripts_dir = project_root / "scripts"
        
        self.menu_structure = {}
        self.script_calls = []
        self.available_scripts = []
        
    def audit(self) -> Dict:
        """Run full audit."""
        print("="*80)
        print("KINETRA MENU SYSTEM AUDIT")
        print("="*80)
        
        # Step 1: Parse menu structure
        print("\n[1] Parsing menu structure...")
        self.menu_structure = self._parse_menu_structure()
        print(f"    Found {len(self.menu_structure)} menu functions")
        
        # Step 2: Extract script calls
        print("\n[2] Extracting script calls from menu...")
        self.script_calls = self._extract_script_calls()
        print(f"    Found {len(self.script_calls)} script calls")
        
        # Step 3: Find all available scripts
        print("\n[3] Scanning available scripts...")
        self.available_scripts = self._find_available_scripts()
        print(f"    Found {len(self.available_scripts)} total scripts")
        
        # Step 4: Analyze coverage
        print("\n[4] Analyzing coverage...")
        analysis = self._analyze_coverage()
        
        return analysis
        
    def _parse_menu_structure(self) -> Dict:
        """Parse menu structure from kinetra_menu.py."""
        with open(self.menu_file, 'r') as f:
            content = f.read()
            
        # Find all menu/run functions
        pattern = r'def (show_|run_)(\w+)\(.*?\):'
        matches = re.findall(pattern, content)
        
        menu_functions = {}
        for prefix, name in matches:
            func_name = f"{prefix}{name}"
            
            # Extract docstring
            doc_pattern = rf'def {func_name}\(.*?\):\s+"""(.*?)"""'
            doc_match = re.search(doc_pattern, content, re.DOTALL)
            description = doc_match.group(1).strip() if doc_match else "No description"
            
            menu_functions[func_name] = {
                'description': description.split('\n')[0],  # First line only
                'type': 'menu' if prefix == 'show_' else 'action'
            }
            
        return menu_functions
        
    def _extract_script_calls(self) -> List[Dict]:
        """Extract all subprocess.run calls to scripts."""
        with open(self.menu_file, 'r') as f:
            content = f.read()
            
        # Find subprocess.run calls
        pattern = r'subprocess\.run\(\s*\[\s*sys\.executable,\s*["\']([^"\']+)["\']\]?'
        matches = re.findall(pattern, content)
        
        script_calls = []
        for script_path in matches:
            # Find which function calls this script
            func_pattern = rf'def (\w+)\(.*?\):.*?{re.escape(script_path)}'
            func_match = re.search(func_pattern, content, re.DOTALL)
            caller = func_match.group(1) if func_match else "unknown"
            
            script_calls.append({
                'script': script_path,
                'called_by': caller,
                'exists': (self.project_root / script_path).exists()
            })
            
        return script_calls
        
    def _find_available_scripts(self) -> List[Dict]:
        """Find all Python scripts in the project."""
        scripts = []
        
        for script_file in self.scripts_dir.glob("**/*.py"):
            rel_path = script_file.relative_to(self.project_root)
            
            # Check if it's called by menu
            called = any(call['script'] == str(rel_path) for call in self.script_calls)
            
            # Check if it's a library/module (in subdirectory with __init__.py)
            parent_has_init = (script_file.parent / "__init__.py").exists()
            
            scripts.append({
                'path': str(rel_path),
                'name': script_file.stem,
                'category': script_file.parent.name,
                'called_by_menu': called,
                'is_module': parent_has_init
            })
            
        return scripts
        
    def _analyze_coverage(self) -> Dict:
        """Analyze which scripts are used/unused."""
        # Scripts called by menu
        called_scripts = {call['script'] for call in self.script_calls}
        
        # All available scripts (excluding modules)
        all_scripts = {s['path'] for s in self.available_scripts if not s['is_module']}
        
        # Find deadweight
        unused_scripts = all_scripts - called_scripts
        
        # Find missing scripts (called but don't exist)
        missing_scripts = [call for call in self.script_calls if not call['exists']]
        
        # Group by category
        by_category = defaultdict(lambda: {'called': [], 'unused': []})
        for script in self.available_scripts:
            if script['is_module']:
                continue
                
            category = script['category']
            if script['called_by_menu']:
                by_category[category]['called'].append(script['name'])
            else:
                by_category[category]['unused'].append(script['name'])
                
        return {
            'total_menu_functions': len(self.menu_structure),
            'total_script_calls': len(self.script_calls),
            'total_available_scripts': len(all_scripts),
            'called_scripts': len(called_scripts),
            'unused_scripts': len(unused_scripts),
            'missing_scripts': len(missing_scripts),
            'by_category': dict(by_category),
            'unused_script_list': sorted(unused_scripts),
            'missing_script_list': [s['script'] for s in missing_scripts],
        }
        
    def generate_report(self, analysis: Dict) -> str:
        """Generate comprehensive audit report."""
        report = f"""
{'='*80}
MENU SYSTEM AUDIT REPORT
{'='*80}

📊 SUMMARY
{'='*80}

Total Menu Functions:    {analysis['total_menu_functions']}
Total Script Calls:      {analysis['total_script_calls']}
Available Scripts:       {analysis['total_available_scripts']}
Scripts Called by Menu:  {analysis['called_scripts']}
Unused Scripts:          {analysis['unused_scripts']}
Missing Scripts:         {analysis['missing_scripts']}

Coverage: {analysis['called_scripts']/analysis['total_available_scripts']*100:.1f}%

{'='*80}
MENU STRUCTURE
{'='*80}

"""
        
        # Menu functions by type
        menus = {k: v for k, v in self.menu_structure.items() if v['type'] == 'menu'}
        actions = {k: v for k, v in self.menu_structure.items() if v['type'] == 'action'}
        
        report += f"Menu Functions ({len(menus)}):\n"
        for name, info in sorted(menus.items()):
            report += f"  • {name}: {info['description']}\n"
            
        report += f"\nAction Functions ({len(actions)}):\n"
        for name, info in sorted(actions.items()):
            report += f"  • {name}: {info['description']}\n"
            
        # Script calls
        report += f"\n{'='*80}\nSCRIPT CALLS (Menu → Scripts)\n{'='*80}\n\n"
        
        for call in sorted(self.script_calls, key=lambda x: x['called_by']):
            exists_icon = "✓" if call['exists'] else "✗"
            report += f"  [{exists_icon}] {call['called_by']} → {call['script']}\n"
            
        # Coverage by category
        report += f"\n{'='*80}\nCOVERAGE BY CATEGORY\n{'='*80}\n\n"
        
        for category, scripts in sorted(analysis['by_category'].items()):
            total = len(scripts['called']) + len(scripts['unused'])
            called_pct = len(scripts['called']) / total * 100 if total > 0 else 0
            
            report += f"{category}/:\n"
            report += f"  Total:  {total}\n"
            report += f"  Called: {len(scripts['called'])} ({called_pct:.0f}%)\n"
            report += f"  Unused: {len(scripts['unused'])}\n"
            
            if scripts['called']:
                report += f"  ✓ Used: {', '.join(sorted(scripts['called']))}\n"
            if scripts['unused']:
                report += f"  ✗ Unused: {', '.join(sorted(scripts['unused']))}\n"
            report += "\n"
            
        # Deadweight details
        if analysis['unused_scripts']:
            report += f"{'='*80}\n⚠️  DEADWEIGHT (Unused Scripts)\n{'='*80}\n\n"
            report += "These scripts exist but are not called by the menu:\n\n"
            
            for script in sorted(analysis['unused_script_list']):
                report += f"  • {script}\n"
                
            report += "\nRecommendation: Review if these should be:\n"
            report += "  1. Integrated into menu system\n"
            report += "  2. Documented as standalone tools\n"
            report += "  3. Removed as obsolete\n"
            
        # Missing scripts
        if analysis['missing_scripts']:
            report += f"\n{'='*80}\n❌ MISSING SCRIPTS (Called but don't exist)\n{'='*80}\n\n"
            
            for script in analysis['missing_script_list']:
                report += f"  • {script}\n"
                
            report += "\nRecommendation: Either:\n"
            report += "  1. Implement these scripts\n"
            report += "  2. Remove calls from menu\n"
            report += "  3. Replace with placeholders\n"
            
        # Recommendations
        report += f"\n{'='*80}\n💡 RECOMMENDATIONS\n{'='*80}\n\n"
        
        if analysis['unused_scripts'] > 10:
            report += "⚠️  HIGH DEADWEIGHT: Consider cleanup\n"
        if analysis['missing_scripts'] > 0:
            report += "❌ BROKEN LINKS: Fix missing script calls\n"
        if analysis['called_scripts'] / analysis['total_available_scripts'] < 0.5:
            report += "📊 LOW COVERAGE: Many scripts not integrated\n"
            
        report += "\n"
        
        return report
        
    def generate_visual_map(self) -> str:
        """Generate visual menu → script mapping."""
        map_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                        KINETRA MENU SYSTEM MAP                             ║
╚════════════════════════════════════════════════════════════════════════════╝

MAIN MENU
│
├─[1] Login & Authentication
│   ├─ Select MetaAPI Account → scripts/download/select_metaapi_account.py
│   └─ Test Connection (inline)
│
├─[2] Exploration Testing
│   ├─ Quick Exploration → run_comprehensive_exploration.py
│   ├─ Custom Exploration → run_comprehensive_exploration.py
│   ├─ Scientific Discovery → scripts/testing/run_scientific_testing.py
│   ├─ Agent Comparison → scripts/training/explore_compare_agents.py
│   └─ Measurement Analysis (NOT IMPLEMENTED)
│
├─[3] Backtesting
│   ├─ Quick Backtest → scripts/testing/run_comprehensive_backtest.py
│   ├─ Custom Backtesting → scripts/testing/run_comprehensive_backtest.py
│   ├─ Monte Carlo → scripts/testing/run_comprehensive_backtest.py
│   ├─ Walk-Forward (NOT IMPLEMENTED)
│   └─ Comparative Analysis (NOT IMPLEMENTED)
│
├─[4] Data Management
│   ├─ Auto-Download (inline)
│   ├─ Manual Download → scripts/download/download_interactive.py
│   ├─ Check & Fill → scripts/download/check_and_fill_data.py
│   ├─ Data Integrity → scripts/download/check_data_integrity.py
│   ├─ Prepare Data → scripts/download/prepare_data.py
│   └─ Backup & Restore → scripts/download/backup_data.py
│
└─[5] System Status & Health
    ├─ Current Health (inline)
    ├─ Recent Results (inline)
    ├─ Data Summary (inline)
    └─ Performance Metrics (inline)

Legend:
  → Script call
  (inline) = Implemented directly in menu
  (NOT IMPLEMENTED) = Placeholder, shows warning
"""
        return map_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run menu audit."""
    project_root = Path(__file__).parent.parent
    
    auditor = MenuAuditor(project_root)
    analysis = auditor.audit()
    
    # Generate report
    report = auditor.generate_report(analysis)
    print(report)
    
    # Generate visual map
    visual_map = auditor.generate_visual_map()
    print(visual_map)
    
    # Save reports
    output_dir = project_root / "docs"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "MENU_AUDIT_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
        f.write("\n\n")
        f.write(visual_map)
        
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Save JSON for programmatic access
    json_file = output_dir / "menu_audit.json"
    with open(json_file, 'w') as f:
        json.dump(analysis, f, indent=2)
        
    print(f"📄 JSON data saved to: {json_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
