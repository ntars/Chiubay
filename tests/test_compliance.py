"""自研合规扫描：Child 与 runtime 路径禁止出现现成 ML 库。"""
import ast
import unittest
from pathlib import Path

FORBIDDEN = {
    "torch",
    "tensorflow",
    "jax",
    "transformers",
    "sklearn",
    "gymnasium",
    "autograd",
    "keras",
}


class TestCompliance(unittest.TestCase):
    def test_no_forbidden_imports(self):
        root = Path(__file__).resolve().parents[1] / "src"
        violations = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in FORBIDDEN:
                            violations.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] in FORBIDDEN:
                        violations.append(f"{path}: from {node.module}")
        self.assertEqual(violations, [], msg="\n".join(violations))


if __name__ == "__main__":
    unittest.main()