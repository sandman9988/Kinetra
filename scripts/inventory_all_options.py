#!/usr/bin/env python3
"""
COMPLETE SCRIPT OPTIONS INVENTORY
==================================
Scan ALL scripts and extract ALL options, arguments, and capabilities.
"""
import ast
import re
from pathlib import Path

def extract_all_info(filepath):
    """Extract every option, argument, and capability from a script."""
    try:
        content = Path(filepath).read_text()
        
        info = {
            'file': str(filepath),
            'docstring': '',
            'cli_args': [],
            'functions': [],
            'classes': [],
            'imports': [],
            'constants': {},
            'hardcoded_values': []
        }
        
        # Docstring
        doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if doc_match:
            info['docstring'] = doc_match.group(1).strip()[:500]
        
        # CLI arguments (argparse)
        for match in re.finditer(r'add_argument\((.*?)\)', content, re.DOTALL):
            arg_text = match.group(1)
            info['cli_args'].append(arg_text[:200])
        
        # Functions
        for match in re.finditer(r'^def (\w+)\((.*?)\):', content, re.MULTILINE):
            info['functions'].append(f"{match.group(1)}({match.group(2)[:50]})")
        
        # Classes
        for match in re.finditer(r'^class (\w+)', content, re.MULTILINE):
            info['classes'].append(match.group(1))
        
        # Imports
        for match in re.finditer(r'^(?:from|import) ([\w.]+)', content, re.MULTILINE):
            info['imports'].append(match.group(1))
        
        # Constants (UPPER_CASE = value)
        for match in re.finditer(r'^([A-Z_]+)\s*=\s*(.+)$', content, re.MULTILINE):
            info['constants'][match.group(1)] = match.group(2).strip()[:100]
        
        # Hardcoded instrument/timeframe lists
        for pattern in [r'INSTRUMENTS\s*=\s*\[(.*?)\]', r'TIMEFRAMES\s*=\s*\[(.*?)\]', 
                       r'SYMBOLS\s*=\s*\[(.*?)\]', r'ASSET_CLASSES\s*=\s*\[(.*?)\]']:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                info['hardcoded_values'].append(match.group(0)[:200])
        
        return info
    except Exception as e:
        return {'file': str(filepath), 'error': str(e)}

# Scan all Python scripts
scripts_dir = Path('scripts')
all_py_files = sorted(scripts_dir.rglob('*.py'))

output = []
for script in all_py_files:
    if '__pycache__' in str(script) or 'inventory_all_options' in str(script):
        continue
    
    info = extract_all_info(script)
    
    output.append(f"\n{'='*100}")
    output.append(f"FILE: {info['file']}")
    output.append(f"{'='*100}")
    
    if 'error' in info:
        output.append(f"ERROR: {info['error']}")
        continue
    
    if info['docstring']:
        output.append(f"\nDESCRIPTION:\n{info['docstring']}\n")
    
    if info['cli_args']:
        output.append(f"\nCLI ARGUMENTS ({len(info['cli_args'])}):")
        for arg in info['cli_args']:
            output.append(f"  • {arg}")
    
    if info['constants']:
        output.append(f"\nCONSTANTS ({len(info['constants'])}):")
        for name, value in list(info['constants'].items())[:10]:
            output.append(f"  {name} = {value}")
    
    if info['hardcoded_values']:
        output.append(f"\nHARDCODED LISTS:")
        for val in info['hardcoded_values']:
            output.append(f"  {val}")
    
    if info['functions']:
        output.append(f"\nFUNCTIONS ({len(info['functions'])}):")
        for func in info['functions'][:20]:
            output.append(f"  • {func}")
    
    if info['classes']:
        output.append(f"\nCLASSES ({len(info['classes'])}):")
        for cls in info['classes'][:20]:
            output.append(f"  • {cls}")

print('\n'.join(output))
