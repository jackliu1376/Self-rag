import sqlite3
from pathlib import Path

from app.repositories.migrations import run_migrations
from app.repositories.models import Base


def test_run_migrations_creates_every_model_table(tmp_path: Path) -> None:
    db = tmp_path / "migrated.db"
    run_migrations(f"sqlite+aiosqlite:///{db.as_posix()}")
    with sqlite3.connect(db) as connection:
        names = {
            row[0]
            for row in connection.execute("select name from sqlite_master where type='table'")
        }
    assert set(Base.metadata.tables) <= names
    assert "alembic_version" in names
