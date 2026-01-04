#!/usr/bin/env python3
"""
Auto-Documentation Generator
=============================

Automatically generates documentation from code docstrings, type hints, and module structure.

Features:
- Extracts docstrings from modules, classes, functions
- Generates markdown documentation
- Creates API reference
- Updates module index
- Generates dependency graphs

Usage:
    # Generate all documentation
    python scripts/housekeeping/generate_docs.py --all

    # Generate specific module docs
    python scripts/housekeeping/generate_docs.py --module kinetra.physics_engine

    # Generate API reference
    python scripts/housekeeping/generate_docs.py --api-ref

__version__ = "1.0.0"
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from AST node."""
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        docstring = ast.get_docstring(node)
        return docstring
    return None


def get_function_signature(node: ast.FunctionDef) -> str:
    """Extract function signature as string."""
    args = []

    # Regular args
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)

    # Return type
    returns = ""
    if node.returns:
        returns = f" -> {ast.unparse(node.returns)}"

    return f"{node.name}({', '.join(args)}){returns}"


def parse_module(filepath: Path) -> Dict[str, Any]:
    """Parse Python module and extract documentation."""
    source = filepath.read_text()
    tree = ast.parse(source)

    module_doc = ast.get_docstring(tree)

    # Extract version
    version = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if isinstance(node.value, ast.Constant):
                        version = node.value.value

    # Extract classes and functions
    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node)
            methods = []

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_doc = ast.get_docstring(item)
                    methods.append({
                        "name": item.name,
                        "signature": get_function_signature(item),
                        "docstring": method_doc,
                    })

            classes.append({
                "name": node.name,
                "docstring": class_doc,
                "methods": methods,
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_doc = ast.get_docstring(node)
            functions.append({
                "name": node.name,
                "signature": get_function_signature(node),
                "docstring": func_doc,
            })

    return {
        "filepath": filepath,
        "module_doc": module_doc,
        "version": version,
        "classes": classes,
        "functions": functions,
    }


def generate_module_markdown(module_info: Dict[str, Any]) -> str:
    """Generate markdown documentation for a module."""
    md = []

    # Header
    module_name = module_info["filepath"].stem
    md.append(f"# {module_name}")
    md.append("")

    if module_info["version"]:
        md.append(f"**Version:** {module_info['version']}")
        md.append("")

    # Module docstring
    if module_info["module_doc"]:
        md.append(module_info["module_doc"])
        md.append("")

    # Classes
    if module_info["classes"]:
        md.append("## Classes")
        md.append("")

        for cls in module_info["classes"]:
            md.append(f"### `{cls['name']}`")
            md.append("")

            if cls["docstring"]:
                md.append(cls["docstring"])
                md.append("")

            if cls["methods"]:
                md.append("**Methods:**")
                md.append("")

                for method in cls["methods"]:
                    if method["name"].startswith("_"):
                        continue  # Skip private methods

                    md.append(f"#### `{method['signature']}`")
                    md.append("")

                    if method["docstring"]:
                        md.append(method["docstring"])
                        md.append("")

    # Functions
    if module_info["functions"]:
        md.append("## Functions")
        md.append("")

        for func in module_info["functions"]:
            if func["name"].startswith("_"):
                continue  # Skip private functions

            md.append(f"### `{func['signature']}`")
            md.append("")

            if func["docstring"]:
                md.append(func["docstring"])
                md.append("")

    return "\n".join(md)


def scan_package(package_dir: Path) -> List[Path]:
    """Scan package directory for Python modules."""
    modules = []

    for filepath in package_dir.rglob("*.py"):
        # Skip __pycache__, tests, examples
        if any(part.startswith("__") for part in filepath.parts):
            continue
        if "test" in filepath.parts:
            continue
        if "example" in filepath.parts:
            continue

        modules.append(filepath)

    return sorted(modules)


def generate_api_reference(package_dir: Path, output_dir: Path):
    """Generate complete API reference documentation."""
    modules = scan_package(package_dir)

    print(f"📚 Found {len(modules)} modules to document")

    # Generate individual module docs
    for module_path in modules:
        try:
            print(f"  Parsing {module_path.name}...")
            module_info = parse_module(module_path)

            # Generate markdown
            markdown = generate_module_markdown(module_info)

            # Save to docs/api/
            api_dir = output_dir / "api"
            api_dir.mkdir(parents=True, exist_ok=True)

            output_file = api_dir / f"{module_path.stem}.md"
            output_file.write_text(markdown)

            print(f"    ✅ {output_file.relative_to(PROJECT_ROOT)}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")

    # Generate index
    generate_api_index(modules, output_dir / "api")


def generate_api_index(modules: List[Path], api_dir: Path):
    """Generate API index page."""
    md = []

    md.append("# Kinetra API Reference")
    md.append("")
    md.append("Auto-generated API documentation from code docstrings.")
    md.append("")
    md.append("## Modules")
    md.append("")

    for module_path in modules:
        module_name = module_path.stem
        relative_import = ".".join(module_path.relative_to(PROJECT_ROOT / "kinetra").with_suffix("").parts)

        md.append(f"- [{module_name}]({module_name}.md) - `kinetra.{relative_import}`")

    md.append("")
    md.append("---")
    md.append("*Generated automatically by `scripts/housekeeping/generate_docs.py`*")

    index_path = api_dir / "README.md"
    index_path.write_text("\n".join(md))

    print(f"\n✅ Generated API index: {index_path.relative_to(PROJECT_ROOT)}")


def generate_module_index(docs_dir: Path):
    """Generate docs/README.md index."""
    md = []

    md.append("# Kinetra Documentation")
    md.append("")
    md.append("## Quick Start")
    md.append("")
    md.append("- [Installation Guide](../INSTALL.md)")
    md.append("- [Quick Reference](QUICK_REFERENCE_CLI_E2E.md)")
    md.append("- [Scientific Testing Guide](SCIENTIFIC_TESTING_GUIDE.md)")
    md.append("")
    md.append("## Architecture")
    md.append("")
    md.append("- [Architecture Overview](architecture.md)")
    md.append("- [Data Management](DATA_MANAGEMENT_ARCHITECTURE.md)")
    md.append("- [Workflow](WORKFLOW.md)")
    md.append("")
    md.append("## API Reference")
    md.append("")
    md.append("- [API Documentation](api/README.md)")
    md.append("")
    md.append("## Testing")
    md.append("")
    md.append("- [Testing Framework](TESTING_FRAMEWORK.md)")
    md.append("- [Scientific Testing](SCIENTIFIC_TESTING_GUIDE.md)")
    md.append("")
    md.append("## Deployment")
    md.append("")
    md.append("- [DevOps Guide](DEVOPS.md)")
    md.append("- [Deployment](deployment.md)")
    md.append("")
    md.append("---")
    md.append("*For agent rules and development guidelines, see [AGENT_RULES_MASTER.md](../AGENT_RULES_MASTER.md)*")

    index_path = docs_dir / "README.md"
    index_path.write_text("\n".join(md))

    print(f"✅ Generated docs index: {index_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Generate documentation")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all documentation"
    )
    parser.add_argument(
        "--api-ref",
        action="store_true",
        help="Generate API reference"
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Generate docs for specific module"
    )

    args = parser.parse_args()

    package_dir = PROJECT_ROOT / "kinetra"
    docs_dir = PROJECT_ROOT / "docs"

    print("=" * 80)
    print("KINETRA DOCUMENTATION GENERATOR")
    print("=" * 80)
    print()

    if args.all or args.api_ref:
        print("📚 Generating API reference...")
        generate_api_reference(package_dir, docs_dir)
        print()

    if args.all:
        print("📋 Generating documentation index...")
        generate_module_index(docs_dir)
        print()

    if args.module:
        module_path = PROJECT_ROOT / args.module.replace(".", "/") + ".py"
        if module_path.exists():
            print(f"📄 Generating docs for {args.module}...")
            module_info = parse_module(module_path)
            markdown = generate_module_markdown(module_info)
            print(markdown)
        else:
            print(f"❌ Module not found: {module_path}")
            return 1

    if not any([args.all, args.api_ref, args.module]):
        parser.print_help()
        return 1

    print("✅ Documentation generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
