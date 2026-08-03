from __future__ import annotations

import os
from pathlib import Path
import sys

# Allows `python app.py` to work even before an editable installation is refreshed.
BACKEND_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from dotenv import load_dotenv

from multivari.api import create_app

load_dotenv(BACKEND_ROOT / ".env")

app = create_app()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    app.run(host=host, port=port, debug=debug)
