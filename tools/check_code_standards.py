from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

MAXIMUM_FUNCTION_LINES = 10
FUTURE_IMPORT = "from __future__ import annotations"
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
DOCUMENTED_NODES = (ast.Module, ast.ClassDef, *FUNCTION_NODES)


def oversized_functions(path: Path, tree: ast.Module) -> list[str]:
    functions = [node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)]
    lengths = [(node, (node.end_lineno or 0) - node.body[0].lineno + 1) for node in functions]
    return [
        f"{path}:{node.lineno}: '{node.name}' has {length} lines (max {MAXIMUM_FUNCTION_LINES})"
        for node, length in lengths
        if length > MAXIMUM_FUNCTION_LINES
    ]


def comments(path: Path, source: str) -> list[str]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    found = [token for token in tokens if token.type == tokenize.COMMENT]
    return [f"{path}:{token.start[0]}: comments are not allowed" for token in found]


def docstrings(path: Path, tree: ast.Module) -> list[str]:
    owners = [node for node in ast.walk(tree) if isinstance(node, DOCUMENTED_NODES)]
    documented = [node for node in owners if ast.get_docstring(node) is not None]
    return [
        f"{path}:{getattr(node, 'lineno', 1)}: docstrings are not allowed" for node in documented
    ]


def missing_future_import(path: Path, source: str) -> list[str]:
    if not source.strip() or source.startswith(FUTURE_IMPORT):
        return []
    return [f"{path}:1: module must start with '{FUTURE_IMPORT}'"]


def violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    structural = oversized_functions(path, tree) + docstrings(path, tree)
    return structural + comments(path, source) + missing_future_import(path, source)


def main(arguments: list[str]) -> int:
    found = [violation for name in arguments for violation in violations(Path(name))]
    for violation in found:
        sys.stdout.write(violation + "\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
