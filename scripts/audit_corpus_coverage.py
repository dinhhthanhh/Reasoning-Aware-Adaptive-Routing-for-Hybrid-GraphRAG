"""Corpus coverage audit — thin wrapper for promt.md Phase 0.0.

    python scripts/audit_corpus_coverage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase0_audit import main  # noqa: E402

if __name__ == "__main__":
    main()
