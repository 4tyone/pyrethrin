from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def _ast_to_json(node: ast.AST | None) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.AST):
        result: dict[str, Any] = {"_type": node.__class__.__name__}
        for field_name, value in ast.iter_fields(node):
            result[field_name] = _ast_to_json(value)
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                result[attr] = getattr(node, attr)
        return result
    elif isinstance(node, list):
        return [_ast_to_json(item) for item in node]
    elif isinstance(node, (str, int, float, bool, type(None))):
        return node
    elif isinstance(node, bytes):
        return node.decode("utf-8", errors="replace")
    else:
        return str(node)


def dump_raw_ast(
    file_path: str | Path,
    external_signatures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    file_path = Path(file_path)
    source = file_path.read_text()
    tree = ast.parse(source)
    result = {
        "language": "python",
        "source_file": str(file_path),
        "ast": _ast_to_json(tree),
    }
    if external_signatures:
        result["external_signatures"] = external_signatures
    return result


def dump_raw_ast_json(
    file_path: str | Path,
    indent: int | None = 2,
    external_signatures: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(dump_raw_ast(file_path, external_signatures), indent=indent)
