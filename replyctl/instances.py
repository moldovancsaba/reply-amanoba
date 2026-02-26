from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import secrets
import signal
import subprocess
import tarfile
import time

from replyctl.manifest import load_manifest, write_manifest
from replyctl.migrations import apply_pending_migrations
from replyctl.models import CompanyManifest, InstancePaths


ROOT_DEFAULT = Path.home() / ".reply-amanoba" / "instances"


def core_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def core_version() -> str:
    version_file = core_repo_root() / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip() or "0.1.0"
    return "0.1.0"


def instance_paths(slug: str, root: Path | None = None) -> InstancePaths:
    base = (root or ROOT_DEFAULT).expanduser().resolve() / slug
    return InstancePaths(
        root=str(base),
        config_dir=str(base / "config"),
        data_dir=str(base / "data"),
        docs_dir=str(base / "data" / "docs"),
        exports_dir=str(base / "data" / "exports"),
        logs_dir=str(base / "logs"),
        run_dir=str(base / "run"),
        backups_dir=str(base / "backups"),
        env_file=str(base / "config" / ".env"),
        manifest_file=str(base / "config" / "company.yaml"),
        state_file=str(base / "config" / "state.json"),
        db_file=str(base / "data" / "helpbot.sqlite3"),
        api_log_file=str(base / "logs" / "api.log"),
        tunnel_log_file=str(base / "logs" / "tunnel.log"),
        api_pid_file=str(base / "run" / "api.pid"),
        tunnel_pid_file=str(base / "run" / "tunnel.pid"),
    )


def _mkdirs(paths: InstancePaths) -> None:
    for p in [
        paths.root,
        paths.config_dir,
        paths.data_dir,
        paths.docs_dir,
        paths.exports_dir,
        paths.logs_dir,
        paths.run_dir,
        paths.backups_dir,
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def _write_env(paths: InstancePaths, manifest: CompanyManifest) -> None:
    token = secrets.token_urlsafe(24)
    env = {
        "MODEL_PROVIDER": "ollama",
        "OLLAMA_URL": manifest.ollama_url,
        "OLLAMA_MODEL": manifest.ollama_model,
        "EMBED_MODEL": manifest.embed_model,
        "DB_PATH": paths.db_file,
        "DOCS_PATH": paths.docs_dir,
        "EXPORTS_PATH": paths.exports_dir,
        "LANGUAGE_PRIMARY": manifest.language_primary,
        "LANGUAGE_ALLOWED": ",".join(manifest.language_allowed),
        "CHAT_API_TOKEN": token,
        "CHAT_ALLOWED_ORIGINS": ",".join(manifest.chat_allowed_origins),
        "CHAT_RATE_LIMIT_PER_MIN": str(manifest.chat_rate_limit_per_minute),
        "CHAT_ENABLED": "1" if manifest.webchat_enabled else "0",
    }
    lines = [f"{k}={v}" for k, v in env.items()]
    Path(paths.env_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _stop_pid(path: Path) -> bool:
    pid = _read_pid(path)
    if not pid:
        return False
    if not _pid_alive(pid):
        path.unlink(missing_ok=True)
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not _pid_alive(pid):
            break
        time.sleep(0.2)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
    path.unlink(missing_ok=True)
    return True


def install_instance(manifest_path: str, instance_root: str | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    paths = instance_paths(manifest.company_slug, Path(instance_root) if instance_root else None)
    _mkdirs(paths)
    write_manifest(paths.manifest_file, manifest)
    _write_env(paths, manifest)

    state = apply_pending_migrations(Path(paths.root), Path(paths.state_file), core_version())

    return {
        "status": "installed",
        "slug": manifest.company_slug,
        "paths": asdict(paths),
        "state": state,
    }


def load_installed_manifest(slug: str, instance_root: str | None = None) -> CompanyManifest:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    return load_manifest(paths.manifest_file)


def start_instance(slug: str, instance_root: str | None = None, public: bool = False) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    manifest = load_manifest(paths.manifest_file)

    api_pid_file = Path(paths.api_pid_file)
    existing_pid = _read_pid(api_pid_file)
    if existing_pid and _pid_alive(existing_pid):
        return {"status": "already_running", "api_pid": existing_pid, "slug": slug}

    env = os.environ.copy()
    env_file = Path(paths.env_file)
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

    api_log = open(paths.api_log_file, "a", encoding="utf-8")
    cmd = [
        "python3",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        manifest.api_host,
        "--port",
        str(manifest.api_port),
    ]
    proc = subprocess.Popen(cmd, cwd=str(core_repo_root()), env=env, stdout=api_log, stderr=api_log)
    api_pid_file.write_text(str(proc.pid), encoding="utf-8")

    tunnel_pid = None
    if public and manifest.public_tunnel_enabled:
        tunnel_log = open(paths.tunnel_log_file, "a", encoding="utf-8")
        tcmd = ["cloudflared", "tunnel", "--url", f"http://{manifest.api_host}:{manifest.api_port}", "--no-autoupdate"]
        tproc = subprocess.Popen(tcmd, cwd=str(core_repo_root()), env=env, stdout=tunnel_log, stderr=tunnel_log)
        Path(paths.tunnel_pid_file).write_text(str(tproc.pid), encoding="utf-8")
        tunnel_pid = tproc.pid

    return {
        "status": "started",
        "slug": slug,
        "api_pid": proc.pid,
        "tunnel_pid": tunnel_pid,
        "url": f"http://{manifest.api_host}:{manifest.api_port}",
    }


def stop_instance(slug: str, instance_root: str | None = None) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    api_stopped = _stop_pid(Path(paths.api_pid_file))
    tunnel_stopped = _stop_pid(Path(paths.tunnel_pid_file))
    return {"status": "stopped", "slug": slug, "api_stopped": api_stopped, "tunnel_stopped": tunnel_stopped}


def status_instance(slug: str, instance_root: str | None = None) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    manifest = load_manifest(paths.manifest_file)
    api_pid = _read_pid(Path(paths.api_pid_file))
    tunnel_pid = _read_pid(Path(paths.tunnel_pid_file))
    return {
        "slug": slug,
        "api_running": bool(api_pid and _pid_alive(api_pid)),
        "api_pid": api_pid,
        "tunnel_running": bool(tunnel_pid and _pid_alive(tunnel_pid)),
        "tunnel_pid": tunnel_pid,
        "url": f"http://{manifest.api_host}:{manifest.api_port}",
        "paths": asdict(paths),
    }


def list_instances(instance_root: str | None = None) -> list[str]:
    root = (Path(instance_root) if instance_root else ROOT_DEFAULT).expanduser().resolve()
    if not root.exists():
        return []
    slugs = []
    for p in sorted(root.glob("*")):
        if not p.is_dir():
            continue
        if (p / "config" / "company.yaml").exists():
            slugs.append(p.name)
    return slugs


def uninstall_instance(slug: str, instance_root: str | None = None) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    stop_instance(slug, instance_root)
    root = Path(paths.root)
    if root.exists():
        for p in sorted(root.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            root.rmdir()
        except OSError:
            pass
    return {"status": "uninstalled", "slug": slug}


def backup_instance(slug: str, instance_root: str | None = None) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    root = Path(paths.root)
    if not root.exists():
        raise ValueError(f"Instance not found: {slug}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = Path(paths.backups_dir) / f"{slug}-{stamp}.tar.gz"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(root, arcname=slug)
    return {"status": "ok", "slug": slug, "backup": str(backup_path)}


def restore_instance(slug: str, backup_path: str, instance_root: str | None = None) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    stop_instance(slug, instance_root)

    root = Path(paths.root)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        uninstall_instance(slug, instance_root)

    backup = Path(backup_path).expanduser().resolve()
    if not backup.exists() or not backup.is_file():
        raise ValueError(f"Backup not found: {backup}")

    with tarfile.open(backup, "r:gz") as tar:
        tar.extractall(path=root.parent)

    return {"status": "restored", "slug": slug, "backup": str(backup)}


def update_instance(slug: str, instance_root: str | None = None) -> dict:
    paths = instance_paths(slug, Path(instance_root) if instance_root else None)
    backup = backup_instance(slug, instance_root)
    state_file = Path(paths.state_file)
    root = Path(paths.root)

    try:
        state = apply_pending_migrations(root, state_file, core_version())
    except Exception as exc:
        restore_instance(slug, backup["backup"], instance_root)
        return {
            "status": "rolled_back",
            "slug": slug,
            "backup": backup["backup"],
            "reason": str(exc),
        }

    return {
        "status": "updated",
        "slug": slug,
        "backup": backup["backup"],
        "state": state,
    }


def update_all_instances(instance_root: str | None = None) -> dict:
    results = []
    for slug in list_instances(instance_root):
        results.append(update_instance(slug, instance_root))
    return {"status": "ok", "count": len(results), "results": results}
