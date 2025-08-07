"""Configuration management for LeanVibe Agent Hive 2.0."""

import os
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class EnvironmentType(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class CLIToolType(str, Enum):
    """Supported CLI agentic coding tools."""

    OPENCODE = "opencode"
    CLAUDE = "claude"
    GEMINI = "gemini"


class EmbeddingProvider(str, Enum):
    """Embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT, env="ENVIRONMENT"
    )
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # CLI Tool Configuration
    preferred_cli_tool: CLIToolType | None = Field(
        default=None, env="PREFERRED_CLI_TOOL"
    )
    cli_tool_timeout: int = Field(default=300, env="CLI_TOOL_TIMEOUT")

    # API Keys (Optional - for fallback)
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    google_ai_api_key: str | None = Field(default=None, env="GOOGLE_AI_API_KEY")

    # Database Configuration
    database_url: str = Field(
        default="postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive",
        env="DATABASE_URL",
    )
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6381", env="REDIS_URL")
    redis_password: str | None = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # Embedding Configuration
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.SENTENCE_TRANSFORMERS, env="EMBEDDING_PROVIDER"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", env="OLLAMA_BASE_URL"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", env="OLLAMA_EMBEDDING_MODEL"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL"
    )

    # Application Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Agent Configuration
    max_concurrent_agents: int = Field(default=50, env="MAX_CONCURRENT_AGENTS")
    agent_heartbeat_interval: int = Field(default=30, env="AGENT_HEARTBEAT_INTERVAL")
    agent_task_timeout: int = Field(default=300, env="AGENT_TASK_TIMEOUT")
    max_workers: int = Field(default=10, env="MAX_WORKERS")

    # Task Queue Configuration
    task_queue_prefix: str = Field(default="hive:queue", env="TASK_QUEUE_PREFIX")
    task_retry_delay_base: int = Field(default=2, env="TASK_RETRY_DELAY_BASE")
    task_max_retries: int = Field(default=3, env="TASK_MAX_RETRIES")
    task_cleanup_interval: int = Field(default=300, env="TASK_CLEANUP_INTERVAL")

    # Message Broker Configuration
    message_retention_hours: int = Field(
        default=168, env="MESSAGE_RETENTION_HOURS"
    )  # 7 days
    message_history_limit: int = Field(default=1000, env="MESSAGE_HISTORY_LIMIT")
    message_batch_size: int = Field(default=100, env="MESSAGE_BATCH_SIZE")

    # Context Engine Configuration
    context_compression_threshold: int = Field(
        default=5, env="CONTEXT_COMPRESSION_THRESHOLD"
    )
    context_importance_decay_days: int = Field(
        default=30, env="CONTEXT_IMPORTANCE_DECAY_DAYS"
    )
    context_max_storage_mb: int = Field(default=1000, env="CONTEXT_MAX_STORAGE_MB")

    # Self-Improvement Configuration
    self_improvement_enabled: bool = Field(default=True, env="SELF_IMPROVEMENT_ENABLED")
    enable_self_modification: bool = Field(default=True, env="ENABLE_SELF_MODIFICATION")
    improvement_proposal_threshold: float = Field(
        default=0.1, env="IMPROVEMENT_PROPOSAL_THRESHOLD"
    )
    code_review_required: bool = Field(default=True, env="CODE_REVIEW_REQUIRED")
    auto_deploy_enabled: bool = Field(default=False, env="AUTO_DEPLOY_ENABLED")

    # Sleep-Wake Configuration
    sleep_enabled: bool = Field(default=True, env="SLEEP_ENABLED")
    sleep_start_hour: int = Field(default=2, env="SLEEP_START_HOUR")  # 2 AM
    sleep_end_hour: int = Field(default=4, env="SLEEP_END_HOUR")  # 4 AM
    sleep_timezone: str = Field(default="UTC", env="SLEEP_TIMEZONE")

    # Security Configuration
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    health_check_interval: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")

    # File System Configuration
    project_root: str = Field(default=os.getcwd(), env="PROJECT_ROOT")
    workspace_dir: str = Field(default="workspace", env="WORKSPACE_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    temp_dir: str = Field(default="tmp", env="TEMP_DIR")

    # Git Configuration
    git_enabled: bool = Field(default=True, env="GIT_ENABLED")
    git_auto_commit: bool = Field(default=True, env="GIT_AUTO_COMMIT")
    git_branch_prefix: str = Field(default="agent/", env="GIT_BRANCH_PREFIX")

    # Development Configuration
    development_mode: bool = Field(default=False, env="DEVELOPMENT_MODE")
    reload_on_change: bool = Field(default=False, env="RELOAD_ON_CHANGE")
    enable_debug_endpoints: bool = Field(default=False, env="ENABLE_DEBUG_ENDPOINTS")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == EnvironmentType.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == EnvironmentType.PRODUCTION

    @property
    def database_config(self) -> dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.debug and self.is_development,
        }

    @property
    def redis_config(self) -> dict[str, Any]:
        """Get Redis configuration."""
        config = {"url": self.redis_url, "db": self.redis_db, "decode_responses": True}
        if self.redis_password:
            config["password"] = self.redis_password
        return config

    @property
    def cli_tools_config(self) -> dict[str, Any]:
        """Get CLI tools configuration."""
        return {
            "preferred_tool": self.preferred_cli_tool,
            "timeout": self.cli_tool_timeout,
            "api_keys": {
                "anthropic": self.anthropic_api_key,
                "openai": self.openai_api_key,
                "google": self.google_ai_api_key,
            },
        }

    @property
    def embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration."""
        return {
            "provider": self.embedding_provider,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_embedding_model,
            "openai_model": self.openai_embedding_model,
        }

    def get_tmux_session_name(self, agent_name: str) -> str:
        """Get tmux session name for an agent."""
        return f"hive-{agent_name}"

    def get_log_file_path(self, component: str) -> str:
        """Get log file path for a component."""
        os.makedirs(self.logs_dir, exist_ok=True)
        return os.path.join(self.logs_dir, f"{component}.log")

    def get_workspace_path(self, subpath: str = "") -> str:
        """Get workspace path."""
        workspace = os.path.join(self.project_root, self.workspace_dir)
        os.makedirs(workspace, exist_ok=True)
        return os.path.join(workspace, subpath) if subpath else workspace

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check required settings for production
        if self.is_production:
            if self.secret_key == "change-me-in-production":
                issues.append("SECRET_KEY must be changed in production")

            if not self.anthropic_api_key and not self.openai_api_key:
                issues.append(
                    "At least one API key (Anthropic or OpenAI) should be set in production"
                )

        # Check directory permissions
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.get_workspace_path(), exist_ok=True)
        except PermissionError:
            issues.append(f"Cannot create directories in {self.project_root}")

        # Check database URL format
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            issues.append("DATABASE_URL must be a PostgreSQL URL")

        # Check Redis URL format
        if not self.redis_url.startswith("redis://"):
            issues.append("REDIS_URL must be a Redis URL")

        return issues


# Global settings instance
settings = Settings()


# Configuration validation function
def validate_environment() -> None:
    """Validate environment configuration and raise if invalid."""
    issues = settings.validate_configuration()
    if issues:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"- {issue}" for issue in issues
        )
        raise ValueError(error_msg)


# Helper functions for common configuration patterns
def get_database_url() -> str:
    """Get database URL."""
    return settings.database_url


def get_redis_url() -> str:
    """Get Redis URL."""
    return settings.redis_url


def get_cli_tool_preference() -> CLIToolType | None:
    """Get preferred CLI tool."""
    return settings.preferred_cli_tool


def is_development_mode() -> bool:
    """Check if in development mode."""
    return settings.is_development


def get_agent_config() -> dict[str, Any]:
    """Get agent configuration."""
    return {
        "max_concurrent": settings.max_concurrent_agents,
        "heartbeat_interval": settings.agent_heartbeat_interval,
        "task_timeout": settings.agent_task_timeout,
        "project_root": settings.project_root,
    }


def get_api_config() -> dict[str, Any]:
    """Get API configuration."""
    return {
        "host": settings.api_host,
        "port": settings.api_port,
        "workers": settings.api_workers,
        "debug": settings.debug,
        "cors_origins": settings.cors_origins,
    }
