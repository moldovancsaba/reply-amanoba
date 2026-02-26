from __future__ import annotations

import argparse
import json

from replyctl.deps import default_adapters
from replyctl.instances import (
    backup_instance,
    install_instance,
    list_instances,
    restore_instance,
    start_instance,
    status_instance,
    stop_instance,
    uninstall_instance,
    update_all_instances,
    update_instance,
)


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="replyctl", description="reply-amanoba instance lifecycle manager")
    parser.add_argument("--instance-root", default=None, help="Override instance root directory")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser("install", help="Install a company instance from company.yaml")
    p_install.add_argument("--manifest", required=True, help="Path to company.yaml")
    p_install.add_argument("--with-deps", action="store_true", help="Install/verify required dependencies first")

    p_start = sub.add_parser("start", help="Start an installed instance")
    p_start.add_argument("slug", help="Instance slug")
    p_start.add_argument("--public", action="store_true", help="Start with public tunnel")

    p_stop = sub.add_parser("stop", help="Stop an installed instance")
    p_stop.add_argument("slug", help="Instance slug")

    p_status = sub.add_parser("status", help="Show instance status")
    p_status.add_argument("slug", nargs="?", default=None, help="Instance slug")

    p_uninstall = sub.add_parser("uninstall", help="Uninstall an instance")
    p_uninstall.add_argument("slug", help="Instance slug")
    p_uninstall.add_argument("--yes", action="store_true", help="Confirm uninstall")

    p_backup = sub.add_parser("backup", help="Create instance backup")
    p_backup.add_argument("slug", help="Instance slug")

    p_restore = sub.add_parser("restore", help="Restore instance from backup")
    p_restore.add_argument("slug", help="Instance slug")
    p_restore.add_argument("--backup", required=True, help="Backup tar.gz path")

    p_update = sub.add_parser("update", help="Update instance or all instances")
    p_update.add_argument("slug", nargs="?", default=None, help="Instance slug")
    p_update.add_argument("--all", action="store_true", help="Update all instances")

    sub.add_parser("doctor", help="Check dependency health")
    sub.add_parser("deps-install", help="Install supported dependencies")
    sub.add_parser("deps-upgrade", help="Upgrade supported dependencies")
    sub.add_parser("deps-verify", help="Verify dependency health")

    return parser.parse_args()


def install_or_upgrade_deps(upgrade: bool = False) -> dict:
    rows = []
    for adapter in default_adapters():
        action = "upgrade" if upgrade else "install"
        try:
            if upgrade:
                adapter.upgrade()
            else:
                adapter.install()
            health = adapter.verify()
            rows.append(
                {
                    "name": adapter.name,
                    "action": action,
                    "installed": health.installed,
                    "healthy": health.healthy,
                    "details": health.details,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": adapter.name,
                    "action": action,
                    "installed": False,
                    "healthy": False,
                    "details": str(exc),
                }
            )
    ok = all(r["healthy"] for r in rows if r["name"] == "ollama")
    return {"ok": ok, "dependencies": rows}


def verify_deps() -> dict:
    rows = [a.health().__dict__ for a in default_adapters()]
    ok = all(r["healthy"] for r in rows if r["name"] == "ollama")
    return {"ok": ok, "dependencies": rows}


def main() -> None:
    args = parse_args()
    root = args.instance_root

    if args.cmd == "install":
        if args.with_deps:
            deps = install_or_upgrade_deps(upgrade=False)
            if not deps["ok"]:
                print_json({"status": "error", "step": "dependencies", **deps})
                raise SystemExit(1)
        print_json(install_instance(args.manifest, root))
        return

    if args.cmd == "start":
        print_json(start_instance(args.slug, root, public=args.public))
        return

    if args.cmd == "stop":
        print_json(stop_instance(args.slug, root))
        return

    if args.cmd == "status":
        if args.slug:
            print_json(status_instance(args.slug, root))
            return
        rows = [{"slug": s, **status_instance(s, root)} for s in list_instances(root)]
        print_json({"instances": rows})
        return

    if args.cmd == "uninstall":
        if not args.yes:
            raise SystemExit("Refusing uninstall without --yes")
        print_json(uninstall_instance(args.slug, root))
        return

    if args.cmd == "backup":
        print_json(backup_instance(args.slug, root))
        return

    if args.cmd == "restore":
        print_json(restore_instance(args.slug, args.backup, root))
        return

    if args.cmd == "update":
        if args.all:
            print_json(update_all_instances(root))
        elif args.slug:
            print_json(update_instance(args.slug, root))
        else:
            raise SystemExit("Usage: replyctl update <slug> OR replyctl update --all")
        return

    if args.cmd == "doctor":
        print_json(verify_deps())
        return

    if args.cmd == "deps-install":
        print_json(install_or_upgrade_deps(upgrade=False))
        return

    if args.cmd == "deps-upgrade":
        print_json(install_or_upgrade_deps(upgrade=True))
        return

    if args.cmd == "deps-verify":
        print_json(verify_deps())
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
