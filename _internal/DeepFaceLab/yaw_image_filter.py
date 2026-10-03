"""Metadata-based pose selection, adapted from yangala's aligned-face tool."""

import argparse
import ast
import json
import operator
import random
import shutil
from pathlib import Path

import numpy as np

from DFLIMG import DFLJPG
from facelib import FaceType, LandmarksProcessor

PRESETS = {
    "wide": "abs(x) >= 40 or abs(y) >= 40",
    "pitch-wide": "abs(y) >= 40",
    "pitch-medium": "abs(y) >= 30",
    "yaw-wide": "abs(x) >= 40",
    "front-20-percent": "abs(x) < 20 and abs(y) < 20 and r < 0.2",
    "not-whole-face": "ft != 4",
}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate_condition(expression, variables):
    """Allow only arithmetic, comparisons, boolean operators and abs()."""
    if len(expression) > 500:
        raise ValueError("Pose condition is too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Pose condition is too complex")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            if not np.isfinite(node.value):
                raise ValueError("Pose constants must be finite")
            return node.value
        if isinstance(node, ast.Name) and node.id in ("x", "y", "r", "ft"):
            return variables[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1 and not node.keywords:
            return abs(visit(node.args[0]))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(visit(value) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(visit(value) for value in node.values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for operation, right_node in zip(node.ops, node.comparators):
                if type(operation) not in _COMPARE:
                    raise ValueError("Unsupported pose comparison")
                right = visit(right_node)
                if not _COMPARE[type(operation)](left, right):
                    return False
                left = right
            return True
        raise ValueError("Pose conditions may use only x, y, r, ft, abs and numeric comparisons")

    return bool(visit(tree))


def check(path, condition, rng=None):
    image = DFLJPG.load(path)
    if image is None or not image.has_data():
        return False, 0.0, 0.0
    points = np.asarray(image.get_landmarks(), dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return False, 0.0, 0.0
    pitch, yaw, _ = LandmarksProcessor.estimate_pitch_yaw_roll(points, size=image.get_shape()[1])
    x, y = float(np.degrees(yaw)), float(np.degrees(pitch))
    variables = dict(x=x, y=y, r=(rng or random).random(), ft=int(FaceType.fromString(image.get_face_type())))
    return evaluate_condition(condition, variables), x, y


def select_directory(directory, condition, seed=42):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    # Validate even an empty directory's condition.
    evaluate_condition(condition, dict(x=0.0, y=0.0, r=0.0, ft=4))
    rng = random.Random(seed)
    records = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"):
            selected, yaw, pitch = check(path, condition, rng)
            records.append(dict(name=path.name, selected=selected, yaw=yaw, pitch=pitch))
    return dict(schemaVersion=1, directory=str(directory), condition=condition, seed=seed,
                selectedCount=sum(item["selected"] for item in records), samples=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="DFL aligned pose filtering (yangala)")
    parser.add_argument("directory")
    parser.add_argument("--preset", choices=PRESETS, default="wide")
    parser.add_argument("--condition", help="Use x=yaw, y=pitch, ft=face type, r=seeded random value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Copy selected files into this new directory")
    parser.add_argument("--move", action="store_true", help="Move selected files and write a recovery manifest")
    args = parser.parse_args(argv)
    if args.move and not args.output:
        parser.error("--move requires --output")
    result = select_directory(args.directory, args.condition or PRESETS[args.preset], args.seed)
    if args.output:
        source = Path(result["directory"])
        output = Path(args.output).resolve()
        if output == source or output.is_dir() and any(output.iterdir()):
            raise ValueError("Pose selection output must be empty and different from the source")
        output.mkdir(parents=True, exist_ok=True)
        selected = [item["name"] for item in result["samples"] if item["selected"]]
        if args.move:
            (output / "restore.json").write_text(json.dumps(dict(source=str(source), names=selected), ensure_ascii=False, indent=2), encoding="utf-8")
        operation = shutil.move if args.move else shutil.copy2
        completed = []
        try:
            for name in selected:
                operation(str(source / name), str(output / name))
                completed.append(name)
        except Exception:
            if args.move:
                for name in reversed(completed):
                    shutil.move(str(output / name), str(source / name))
            raise
        result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
