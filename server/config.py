# server/config.py
import os

# Optional: load local .env if python-dotenv is installed; safe if not.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Secrets & DB URL
SECRET = os.getenv("SECRET", "dev_secret_change_me")
# Local dev defaults to SQLite; on Railway set DATABASE_URL to Postgres (async)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")

# Where per-user CSVs and (optional) pickled models are stored
USER_DATA_ROOT = os.getenv("USER_DATA_ROOT", "models")
