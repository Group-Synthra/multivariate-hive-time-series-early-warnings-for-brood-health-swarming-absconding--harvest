from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the swarming IoT database engine only when it is first needed."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    sslmode = os.getenv("DATABASE_SSLMODE", "require")
    return create_engine(
        database_url,
        connect_args={"sslmode": sslmode},
        pool_pre_ping=True,
    )
