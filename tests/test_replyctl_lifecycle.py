import json
import subprocess
import sys
from pathlib import Path


def run_replyctl(args: list[str], cwd: Path) -> dict:
    cmd = [sys.executable, "-m", "replyctl.cli", *args]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip() or "{}")


def test_replyctl_install_update_uninstall(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    instance_root = tmp_path / "instances"
    manifest = tmp_path / "company.yaml"
    manifest.write_text(
        "\n".join(
            [
                'schema_version: "1"',
                "company:",
                '  name: "CI Corp"',
                '  slug: "ci-corp"',
                "language:",
                '  primary: "en"',
                "  allowed:",
                '    - "en"',
                "runtime:",
                '  host: "127.0.0.1"',
                "  port: 8123",
                "models:",
                '  ollama_url: "http://127.0.0.1:11434"',
                '  ollama_model: "qwen2.5:3b"',
                '  embed_model: "intfloat/multilingual-e5-small"',
                "integrations:",
                "  public_tunnel:",
                "    enabled: false",
                '    provider: "cloudflared"',
                "  webchat:",
                "    enabled: true",
                "    allowed_origins:",
                '      - "http://localhost"',
                "    rate_limit_per_minute: 60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    install = run_replyctl(
        ["--instance-root", str(instance_root), "install", "--manifest", str(manifest)],
        repo_root,
    )
    assert install["status"] == "installed"
    assert install["slug"] == "ci-corp"

    status = run_replyctl(["--instance-root", str(instance_root), "status", "ci-corp"], repo_root)
    assert status["slug"] == "ci-corp"
    assert status["api_running"] is False

    update = run_replyctl(["--instance-root", str(instance_root), "update", "ci-corp"], repo_root)
    assert update["status"] in {"updated", "rolled_back"}

    uninstall = run_replyctl(
        ["--instance-root", str(instance_root), "uninstall", "ci-corp", "--yes"],
        repo_root,
    )
    assert uninstall["status"] == "uninstalled"
