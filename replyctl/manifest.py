from __future__ import annotations

from pathlib import Path
import re

import yaml

from replyctl.models import CompanyManifest

SUPPORTED_SCHEMA_VERSION = "1"
LANG_CODE_RE = re.compile(r"^[a-z]{2,5}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:63] or "instance"


def load_manifest(path: str | Path) -> CompanyManifest:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise ValueError(f"Manifest not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest must be a YAML object")

    schema_version = str(raw.get("schema_version", "")).strip()
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version '{schema_version}'. Expected '{SUPPORTED_SCHEMA_VERSION}'."
        )

    company = raw.get("company") or {}
    if not isinstance(company, dict):
        raise ValueError("'company' must be an object")

    company_name = str(company.get("name", "")).strip()
    if len(company_name) < 2:
        raise ValueError("company.name is required")

    company_slug = str(company.get("slug", "")).strip().lower() or _slugify(company_name)
    if not SLUG_RE.match(company_slug):
        raise ValueError("company.slug must match ^[a-z0-9][a-z0-9-]{1,62}$")

    language = raw.get("language") or {}
    if not isinstance(language, dict):
        raise ValueError("'language' must be an object")

    primary = str(language.get("primary", "en")).strip().lower()
    if not LANG_CODE_RE.match(primary):
        raise ValueError("language.primary must be a language code (e.g. en, hu)")

    allowed = language.get("allowed", [primary])
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("language.allowed must be a non-empty list")
    allowed_clean: list[str] = []
    for item in allowed:
        code = str(item).strip().lower()
        if not LANG_CODE_RE.match(code):
            raise ValueError(f"Invalid language.allowed value: {item}")
        if code not in allowed_clean:
            allowed_clean.append(code)
    if primary not in allowed_clean:
        allowed_clean.insert(0, primary)

    runtime = raw.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("'runtime' must be an object")

    host = str(runtime.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(runtime.get("port", 8000))
    if port < 1 or port > 65535:
        raise ValueError("runtime.port must be between 1 and 65535")

    models = raw.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("'models' must be an object")

    ollama_url = str(models.get("ollama_url", "http://127.0.0.1:11434")).strip()
    ollama_model = str(models.get("ollama_model", "qwen2.5:3b")).strip()
    embed_model = str(models.get("embed_model", "intfloat/multilingual-e5-small")).strip()
    if not ollama_model:
        raise ValueError("models.ollama_model is required")

    integrations = raw.get("integrations") or {}
    if not isinstance(integrations, dict):
        raise ValueError("'integrations' must be an object")

    tunnel = integrations.get("public_tunnel") or {}
    if not isinstance(tunnel, dict):
        raise ValueError("integrations.public_tunnel must be an object")

    tunnel_enabled = bool(tunnel.get("enabled", False))
    tunnel_provider = str(tunnel.get("provider", "cloudflared")).strip().lower() or "cloudflared"

    webchat = integrations.get("webchat") or {}
    if not isinstance(webchat, dict):
        raise ValueError("integrations.webchat must be an object")

    webchat_enabled = bool(webchat.get("enabled", True))
    allowed_origins = webchat.get("allowed_origins", ["http://localhost"])
    if not isinstance(allowed_origins, list):
        raise ValueError("integrations.webchat.allowed_origins must be a list")
    chat_allowed_origins = [str(x).strip() for x in allowed_origins if str(x).strip()]
    if not chat_allowed_origins:
        chat_allowed_origins = ["http://localhost"]

    rate_limit = int(webchat.get("rate_limit_per_minute", 60))
    if rate_limit < 1:
        raise ValueError("integrations.webchat.rate_limit_per_minute must be >= 1")

    return CompanyManifest(
        schema_version=schema_version,
        company_name=company_name,
        company_slug=company_slug,
        language_primary=primary,
        language_allowed=allowed_clean,
        api_host=host,
        api_port=port,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        embed_model=embed_model,
        public_tunnel_enabled=tunnel_enabled,
        public_tunnel_provider=tunnel_provider,
        webchat_enabled=webchat_enabled,
        chat_allowed_origins=chat_allowed_origins,
        chat_rate_limit_per_minute=rate_limit,
    )


def write_manifest(path: str | Path, manifest: CompanyManifest) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": manifest.schema_version,
        "company": {"name": manifest.company_name, "slug": manifest.company_slug},
        "language": {"primary": manifest.language_primary, "allowed": manifest.language_allowed},
        "runtime": {"host": manifest.api_host, "port": manifest.api_port},
        "models": {
            "ollama_url": manifest.ollama_url,
            "ollama_model": manifest.ollama_model,
            "embed_model": manifest.embed_model,
        },
        "integrations": {
            "public_tunnel": {
                "enabled": manifest.public_tunnel_enabled,
                "provider": manifest.public_tunnel_provider,
            },
            "webchat": {
                "enabled": manifest.webchat_enabled,
                "allowed_origins": manifest.chat_allowed_origins,
                "rate_limit_per_minute": manifest.chat_rate_limit_per_minute,
            },
        },
    }
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
