from dataclasses import dataclass, field


@dataclass
class CompanyManifest:
    schema_version: str
    company_name: str
    company_slug: str
    language_primary: str = "en"
    language_allowed: list[str] = field(default_factory=lambda: ["en"])
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    embed_model: str = "intfloat/multilingual-e5-small"
    public_tunnel_enabled: bool = False
    public_tunnel_provider: str = "cloudflared"
    webchat_enabled: bool = True
    chat_allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost"])
    chat_rate_limit_per_minute: int = 60


@dataclass
class InstancePaths:
    root: str
    config_dir: str
    data_dir: str
    docs_dir: str
    exports_dir: str
    logs_dir: str
    run_dir: str
    backups_dir: str
    env_file: str
    manifest_file: str
    state_file: str
    db_file: str
    api_log_file: str
    tunnel_log_file: str
    api_pid_file: str
    tunnel_pid_file: str
