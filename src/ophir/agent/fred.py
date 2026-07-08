"""Optional free FRED API key storage for the macro signal.

Relocated out of ``ophir.register`` so the upstream ``register`` package stays
pristine across merges. Reads/writes ``.fred_key`` under the ophir data dir
(``register.layout.OPHIR_DIR``). The key backs the credit / financial-conditions
/ yield-curve series in :mod:`ophir.agent.macro`; without it that signal falls
back to its VIX-only path.
"""

from __future__ import annotations

import os


def _fred_key_path() -> str:
    """Return the ``.fred_key`` path under the ophir data dir (imported lazily)."""
    from ophir.register.layout import OPHIR_DIR

    return os.path.join(OPHIR_DIR, ".fred_key")


def set_fred_key(key: str) -> None:
    """Persist a free FRED API key (backs the ``ophir fred-key`` command)."""
    with open(_fred_key_path(), "w") as f:
        f.write(f"{key}\n")


def get_fred_key() -> str | None:
    """Return the stored free FRED API key, or ``None`` if not registered."""
    path = _fred_key_path()
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip() or None
