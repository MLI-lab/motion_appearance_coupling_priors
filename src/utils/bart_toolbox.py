"""Location of the BART toolbox (https://github.com/mrirecon/bart, v0.6.00).

Kept free of heavy imports so the driver can resolve the path (and forward it to Ray workers)
without importing BART itself.
"""
import os
from pathlib import Path

# Where setup/user_setup.sh builds BART.
DEFAULT_TOOLBOX_PATH = Path(__file__).resolve().parents[2] / "libs" / "bart-0.6.00"


def toolbox_path() -> str | None:
    """$TOOLBOX_PATH if set, else the setup/user_setup.sh install location if it exists, else None."""
    if os.environ.get("TOOLBOX_PATH"):
        return os.environ["TOOLBOX_PATH"]
    if (DEFAULT_TOOLBOX_PATH / "bart").is_file():
        return str(DEFAULT_TOOLBOX_PATH)
    return None
