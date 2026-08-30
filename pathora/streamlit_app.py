"""Entrypoint for Streamlit Community Cloud.

Point the app at `streamlit_app.py` in the deploy dialog.

Two things have to happen before any pathora import:

1. `src` goes on sys.path. Community Cloud installs requirements.txt but does
   not pip-install the repo itself, so the src-layout package is not importable
   by default.
2. `st.secrets` is copied into the environment. Settings reads env vars via
   pydantic-settings and is lru_cached, so this must run before the first
   `get_settings()` call or the secrets are silently ignored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

# Secrets -> environment, before anything reads Settings.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key.upper(), str(value))
except Exception:  # noqa: BLE001 - no secrets.toml configured is fine
    pass

from pathora.ui.app import main  # noqa: E402

main()
