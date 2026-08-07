from __future__ import annotations

import os
import sys
from pathlib import Path

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

    # Flask debug mode creates a parent process and a reloader child. Start the
    # background poller only in the process that actually serves requests.
    monitor = app.extensions.get("absconding_iot_monitor")
    serving_process = not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if monitor is not None and monitor.enabled and serving_process:
        monitor.start()

    app.run(host=host, port=port, debug=debug)
