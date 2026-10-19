"""Run every layer in order and print the results table plus a bottleneck summary.

Layers are ordered bottom-up and run with -x off, so a failing layer still reports
the layers above it. Use pytest directly for a single layer, e.g.
    python -m pytest tests/test_1_retrieval.py -v
"""

import sys

import pytest


def main() -> int:
    args = ["tests/", "-v", "--tb=short", "-p", "no:cacheprovider"]
    args += sys.argv[1:]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
