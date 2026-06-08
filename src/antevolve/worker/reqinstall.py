"""Installs any required yet missing packages."""
import ast
import argparse
import importlib.util
import json
import os
import sys
import subprocess

from antevolve.filelib import file_ops

_MY_STD_LIBS = {
    'abc',
    'math',
    'random',
    'sys',
    'collections',
    'warnings',
    'sklearn',
    'scipy',
    'numpy',
    'networkx',
    'pandas',
}


def get_imports_from_file(file_path: str):
    """Gets imports from the file using AST."""
    
    if not os.path.exists(file_path):
        return []
    
    tree = ast.parse(file_ops.read_text_file(file_path))
    imports = set()
    for node in ast.walk(tree):
        # Handle 'import x' and 'import x as y'
        if isinstance(node, ast.Import):
            for name in node.names:
                # We only care about the top level package (e.g. 'matplotlib' from 'matplotlib.pyplot')
                top_level_package = name.name.split('.')[0]
                imports.add(top_level_package)

        # Handle 'from x import y'
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # node.module can be None for relative imports like 'from . import x'
                top_level_package = node.module.split('.')[0]
                imports.add(top_level_package)

    return list(imports)


def get_installed_packages() -> list[str]:
    """Gets current list of the installed packages."""
    packages = []
    for dist in importlib.metadata.distributions():
        packages.append({
            "name": dist.metadata["Name"],
            "version": dist.version
        })
    return [p['name'] for p in packages]


def filter_std_lib(packages: list[dict]):
    """
    Attempts to filter out standard library modules using sys.stdlib_module_names (Python 3.10+).
    For older python, it falls back to basic sys.builtin checking.
    """
    std_lib = set()
    
    if sys.version_info >= (3, 10):
        std_lib = sys.stdlib_module_names
    else:
        # Fallback for older Python: combine builtin names with a basic list
        std_lib = set(sys.builtin_module_names) 
        # Note: This fallback isn't perfect for older python as it misses modules like 'os', 'json' 
        # that are in the lib folder but not "builtin". 

    third_party = []
    std_lib = std_lib.union(_MY_STD_LIBS)
    for pkg in packages:
        # Check against standard lib list
        if pkg not in std_lib:
            third_party.append(pkg)
            
    return third_party



def get_imports(file_path: str):
    """
    Detects all import statements (import and from ... import) in a Python file.

    Args:
        file_path (str): The path to the Python file.

    Returns:
        list: A list of dictionaries, where each dictionary represents an import
              and contains 'module' (the imported module name), 'name' (the
              specific name imported, if any), and 'asname' (the alias, if any).
    """
    imports = []
    try:
        tree = ast.parse(file_ops.read_text_file(file_path), filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'name': None,
                        'asname': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else '' # Handle 'from . import name'
                for alias in node.names:
                    imports.append({
                        'module': module_name,
                        'name': alias.name,
                        'asname': alias.asname
                    })
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    return imports

def is_package_installed(package_name: str) -> bool:
    """Checks if package is installable."""
    spec = importlib.util.find_spec(package_name)
    if spec:
        return True
    return False


def install_package(package_name: str) -> bool:
    """Installs package if not exist."""
    if is_package_installed(package_name=package_name):
        return True
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def install_for_file(file_path: str) -> bool:
    """Install all necessary packages for a file."""
    imports = get_imports(file_path)
    unique_imports = list(set([i['module'] for i in imports]))
    for imp in unique_imports:
        importing = imp
        if '.' in imp:
            if 'importlib' not in imp and 'fastapi_utils' not in imp:
                importing = imp.replace('.', '-')
        if 'absl' in importing:
            importing = 'absl-py'
        print('Found import ', importing)
        if not install_package(importing):
            return False
    return True


if __name__ == '__main__':
    d = dict()
    d['constant_prompt'] = ''
    d['stochastic_prompts'] = [
        (0.2, "Propose the crazy idea about this program overall."),
        (0.4, "Propose a crazy change to tool, or propose a completely new tool for the agent."),
        (0.4, "Propose a crazy change to an agent prompt only. Don't change the code at all."),
    ]
    # file_ops.write_json_file('test.json', d)

    parser = argparse.ArgumentParser(description="Flask Secret Revealer with Flag.")
    parser.add_argument(
        '--file', 
        type=str, 
        default=None,
        help="The secret string to be revealed."
    )
    args = parser.parse_args()
    if not args.file:
        print('File must be set.')
    print('Installing for ', args.file)
    install_for_file(args.file)
