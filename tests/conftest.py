"""Fixtures partilhadas para os testes do orchestrator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the orchestrator package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Point config to test directory so orchestrator.toml is found
os.environ.setdefault("ORC_ORCHESTRATOR_PORT", "8585")
