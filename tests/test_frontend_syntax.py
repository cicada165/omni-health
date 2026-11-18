"""
tests/test_frontend_syntax.py — Automated syntax validation for frontend assets.

Guarantees that fatal JavaScript syntax errors, unmatched brackets, or malformed
script tags in scheduler/static/index.html are caught in automated CI/pytest runs
and cannot recur unnoticed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_HTML_PATH = Path(__file__).parent.parent / "scheduler" / "static" / "index.html"


def test_index_html_exists() -> None:
    """Verify that index.html exists and is non-empty."""
    assert INDEX_HTML_PATH.exists(), f"File not found: {INDEX_HTML_PATH}"
    assert INDEX_HTML_PATH.stat().st_size > 1000, "index.html is unexpectedly small or empty"


def test_index_html_javascript_syntax_via_node_check() -> None:
    """
    Extract all inline <script> blocks from index.html and validate their syntax
    using `node --check`.
    Fails immediately if there are syntax errors, unmatched braces, or invalid tokens.
    """
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js is not installed on PATH; skipping node syntax validation.")

    content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s+[^>]*)?>(.*?)</script>", content, re.DOTALL | re.IGNORECASE)

    assert len(scripts) >= 1, "No <script> tags found in index.html"

    for idx, script_content in enumerate(scripts, start=1):
        cleaned_script = script_content.strip()
        if not cleaned_script:
            continue

        res = subprocess.run(
            [node_bin, "--check"],
            input=cleaned_script,
            text=True,
            capture_output=True,
        )

        if res.returncode != 0:
            pytest.fail(
                f"JavaScript syntax error in index.html (script block #{idx}):\n"
                f"Returncode: {res.returncode}\n"
                f"Stderr:\n{res.stderr}"
            )
