"""Fail unless every scorer the ``[score]`` extra provides can run in this environment.

A pin that resolves but doesn't import (pyiqa 0.1.12 on numpy 2, #98) doesn't
crash anything: the scorer reports ``is_available() == False`` and its metric
silently drops out of every report. CI's score-imports job and the ``-score``
image build (#99) run this so that becomes a failure instead.

The heavy packages are imported directly first, so a failure prints the real
traceback rather than only ``available: False``. Safety and pose are not checked:
they need a NudeNet service (``PROOF_NUDENET_URL``) and an argus-puppet service
(``PROOF_PUPPET_URL``), not a package.
"""

from __future__ import annotations

import importlib
import sys

PACKAGES = ("torch", "torchmetrics", "transformers", "pyiqa", "insightface", "onnxruntime", "PIL", "imagehash")
NEEDS_A_SERVICE = {"safety", "pose"}


def main() -> int:
    for name in PACKAGES:
        module = importlib.import_module(name)  # raises with the real traceback
        print(f"import {name} {getattr(module, '__version__', '')}".rstrip())

    from argus_proof.evaluate import all_reporting_scorers

    missing = []
    for scorer in all_reporting_scorers():
        prov = scorer.provenance()
        if prov.metric in NEEDS_A_SERVICE:
            continue
        ok = scorer.is_available()
        print(f"scorer {prov.metric} ({prov.name}): {'available' if ok else 'UNAVAILABLE'}")
        if not ok:
            missing.append(prov.metric)
    if missing:
        print(f"unavailable scorers: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
