from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from pyrethrin._ast_dump import dump_raw_ast


def get_pyrethrum_binary() -> Path | None:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            binary_name = "pyrethrum-darwin-arm64"
        else:
            binary_name = "pyrethrum-darwin-x64"
    elif system == "linux":
        binary_name = "pyrethrum-linux-x64"
    elif system == "windows":
        binary_name = "pyrethrum-windows-x64.exe"
    else:
        return None

    bin_dir = Path(__file__).parent / "bin"
    binary_path = bin_dir / binary_name
    if binary_path.exists():
        return binary_path
    return None


def find_python_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.py"))
    return files


def run_pyrethrum(
    json_data: str,
    format_: str = "text",
    strict: bool = False,
    pyrethrum_path: Path | None = None,
) -> tuple[int, str]:
    if pyrethrum_path is None:
        pyrethrum_path = get_pyrethrum_binary()

    if pyrethrum_path is None or not pyrethrum_path.exists():
        return -1, "Error: Pyrethrum binary not found. Please install pyrethrum."

    cmd = [str(pyrethrum_path), "check", "--stdin", "-f", format_]
    if strict:
        cmd.append("--strict")

    try:
        result = subprocess.run(
            cmd,
            input=json_data,
            capture_output=True,
            text=True,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return result.returncode, output
    except FileNotFoundError:
        return -1, f"Error: Could not execute pyrethrum at {pyrethrum_path}"
    except Exception as e:
        return -1, f"Error running pyrethrum: {e}"


def cmd_check(args: argparse.Namespace) -> int:
    files = find_python_files(args.paths)
    if not files:
        print("No Python files found.", file=sys.stderr)
        return 0

    pyrethrum_path = Path(args.pyrethrum) if args.pyrethrum else get_pyrethrum_binary()

    if pyrethrum_path is None and not args.dump_ast:
        print(
            "Error: Pyrethrum binary not found. "
            "Use --dump-ast to see AST output or --pyrethrum to specify path.",
            file=sys.stderr,
        )
        return 1

    max_returncode = 0
    all_outputs = []

    for f in files:
        try:
            ast_data = dump_raw_ast(f)
            json_data = json.dumps(ast_data)
        except SyntaxError as e:
            print(f"Syntax error in {f}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error analyzing {f}: {e}", file=sys.stderr)
            continue

        if args.dump_ast:
            print(json_data)
            continue

        returncode, output = run_pyrethrum(
            json_data,
            format_=args.format,
            strict=args.strict,
            pyrethrum_path=pyrethrum_path,
        )
        if output and output.strip():
            all_outputs.append(output.strip())
        max_returncode = max(max_returncode, returncode)

    if not args.dump_ast and all_outputs:
        print("\n".join(all_outputs))

    return max_returncode


def cmd_dump(args: argparse.Namespace) -> int:
    files = find_python_files(args.paths)
    if not files:
        print("No Python files found.", file=sys.stderr)
        return 0

    for f in files:
        try:
            ast_data = dump_raw_ast(f)
            print(json.dumps(ast_data, indent=2))
        except SyntaxError as e:
            print(f"Syntax error in {f}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error analyzing {f}: {e}", file=sys.stderr)
            continue

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pyrethrin",
        description="Static analyzer for exhaustive exception handling",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    check_parser = subparsers.add_parser(
        "check",
        help="Check Python files for exhaustive exception handling",
    )
    check_parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to check (default: current directory)",
    )
    check_parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    check_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode (warnings become errors)",
    )
    check_parser.add_argument(
        "--dump-ast",
        action="store_true",
        help="Only dump AST JSON without running pyrethrum",
    )
    check_parser.add_argument(
        "--pyrethrum",
        metavar="PATH",
        help="Path to pyrethrum binary",
    )
    check_parser.set_defaults(func=cmd_check)

    dump_parser = subparsers.add_parser(
        "dump",
        help="Dump AST information as JSON (for debugging)",
    )
    dump_parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to analyze",
    )
    dump_parser.set_defaults(func=cmd_dump)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
