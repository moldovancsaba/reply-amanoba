from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import sys


@dataclass
class DependencyHealth:
    name: str
    installed: bool
    healthy: bool
    details: str


class DependencyAdapter:
    name = "base"

    def install(self) -> None:
        raise NotImplementedError

    def verify(self) -> DependencyHealth:
        raise NotImplementedError

    def upgrade(self) -> None:
        self.install()

    def health(self) -> DependencyHealth:
        return self.verify()


class OllamaAdapter(DependencyAdapter):
    name = "ollama"

    def install(self) -> None:
        if shutil.which("ollama"):
            return
        if sys.platform == "darwin" and shutil.which("brew"):
            subprocess.run(["brew", "install", "ollama"], check=True)
            return
        raise RuntimeError("ollama is not installed and auto-install is unavailable on this OS")

    def verify(self) -> DependencyHealth:
        binary = shutil.which("ollama")
        if not binary:
            return DependencyHealth(self.name, False, False, "ollama binary not found")
        try:
            proc = subprocess.run([binary, "list"], capture_output=True, text=True, timeout=8)
        except Exception as exc:
            return DependencyHealth(self.name, True, False, f"ollama command failed: {exc}")
        healthy = proc.returncode == 0
        details = "ok" if healthy else (proc.stderr.strip() or "failed")
        return DependencyHealth(self.name, True, healthy, details)


class CloudflaredAdapter(DependencyAdapter):
    name = "cloudflared"

    def install(self) -> None:
        if shutil.which("cloudflared"):
            return
        if sys.platform == "darwin" and shutil.which("brew"):
            subprocess.run(["brew", "install", "cloudflared"], check=True)
            return
        raise RuntimeError("cloudflared not installed and auto-install is unavailable on this OS")

    def verify(self) -> DependencyHealth:
        binary = shutil.which("cloudflared")
        if not binary:
            return DependencyHealth(self.name, False, False, "cloudflared binary not found")
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True)
        healthy = proc.returncode == 0
        details = proc.stdout.strip() if healthy else (proc.stderr.strip() or "failed")
        return DependencyHealth(self.name, True, healthy, details)


class OpenClawAdapter(DependencyAdapter):
    name = "openclaw"

    def install(self) -> None:
        # Placeholder adapter: if command does not exist, rely on manual install for now.
        if shutil.which("openclaw"):
            return
        raise RuntimeError("openclaw auto-install not implemented yet")

    def verify(self) -> DependencyHealth:
        binary = shutil.which("openclaw")
        if not binary:
            return DependencyHealth(self.name, False, False, "openclaw binary not found")
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True)
        healthy = proc.returncode == 0
        details = proc.stdout.strip() if healthy else (proc.stderr.strip() or "failed")
        return DependencyHealth(self.name, True, healthy, details)


def default_adapters() -> list[DependencyAdapter]:
    return [OllamaAdapter(), CloudflaredAdapter(), OpenClawAdapter()]


def adapter_by_name(name: str) -> DependencyAdapter:
    lookup = {a.name: a for a in default_adapters()}
    if name not in lookup:
        raise ValueError(f"Unknown adapter: {name}")
    return lookup[name]
