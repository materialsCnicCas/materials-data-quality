"""Data-value quality assessment package."""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Prevent legacy Windows console encodings from breaking status output."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


configure_utf8_stdio()

__version__ = "2.1.0"
