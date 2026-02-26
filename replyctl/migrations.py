from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Migration:
    version: int
    name: str

    def apply(self, instance_root: Path) -> None:
        raise NotImplementedError


class Migration001EnsureExports(Migration):
    def __init__(self) -> None:
        super().__init__(version=1, name="ensure-exports-folder")

    def apply(self, instance_root: Path) -> None:
        (instance_root / "data" / "exports").mkdir(parents=True, exist_ok=True)


class Migration002EnsureArchive(Migration):
    def __init__(self) -> None:
        super().__init__(version=2, name="ensure-qa-archive-folder")

    def apply(self, instance_root: Path) -> None:
        (instance_root / "data" / "docs" / "_qa_archive").mkdir(parents=True, exist_ok=True)


MIGRATIONS: list[Migration] = [
    Migration001EnsureExports(),
    Migration002EnsureArchive(),
]


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"migration_version": 0, "core_version": "0.1.0"}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {"migration_version": 0, "core_version": "0.1.0"}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def apply_pending_migrations(instance_root: Path, state_file: Path, core_version: str) -> dict:
    state = load_state(state_file)
    current = int(state.get("migration_version", 0))

    for m in MIGRATIONS:
        if m.version <= current:
            continue
        m.apply(instance_root)
        current = m.version

    state["migration_version"] = current
    state["core_version"] = core_version
    save_state(state_file, state)
    return state
