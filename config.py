
"""
StarmentAI - Configuration
Centralized application configuration.
"""

import os

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Application configuration."""

    # Application
    app_name: str = "StarmentAI"
    app_version: str = "0.2.0"

    # Ollama connection
    ollama_host: str = "http://localhost:11434"
    ollama_chat_endpoint: str = "/api/chat"

    # Model
    model_name: str = "starmentai"

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 1024
    num_thread: int = 2
    think: bool = False

    # Model memory management
    keep_alive: str = "30s"

    # Request settings
    connect_timeout: int = 10
    read_timeout: int = 300

    # Conversation limits
    max_history_messages: int = 6
    max_user_input_length: int = 2000

    def __post_init__(self) -> None:
        """Validate configuration values."""

        if not self.ollama_host.startswith(("http://", "https://")):
            raise ValueError("Invalid Ollama host URL.")

        if not self.model_name.strip():
            raise ValueError("Model name cannot be empty.")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0 and 2.")

        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("Top_p must be between 0 and 1.")

        if self.num_ctx <= 0:
            raise ValueError("num_ctx must be greater than 0.")

        if self.num_thread <= 0:
            raise ValueError("num_thread must be greater than 0.")

        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("Timeout values must be greater than 0.")

        if self.max_history_messages < 0:
            raise ValueError("History limit cannot be negative.")

        if self.max_user_input_length <= 0:
            raise ValueError("Input length must be greater than 0.")

    @property
    def chat_url(self) -> str:
        """Build Ollama chat endpoint URL."""

        return (
            f"{self.ollama_host.rstrip('/')}/"
            f"{self.ollama_chat_endpoint.lstrip('/')}"
        )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""

        defaults = cls()

        return cls(
            app_name=os.getenv("STARMENTAI_NAME", defaults.app_name),
            app_version=os.getenv("STARMENTAI_VERSION", defaults.app_version),

            ollama_host=os.getenv(
                "OLLAMA_HOST",
                defaults.ollama_host
            ),

            ollama_chat_endpoint=os.getenv(
                "OLLAMA_CHAT_ENDPOINT",
                defaults.ollama_chat_endpoint
            ),

            model_name=os.getenv(
                "OLLAMA_MODEL",
                defaults.model_name
            ),

            temperature=float(os.getenv(
                "OLLAMA_TEMPERATURE",
                defaults.temperature
            )),

            top_p=float(os.getenv(
                "OLLAMA_TOP_P",
                defaults.top_p
            )),

            num_ctx=int(os.getenv(
                "OLLAMA_NUM_CTX",
                defaults.num_ctx
            )),

            num_thread=int(os.getenv(
                "OLLAMA_NUM_THREAD",
                defaults.num_thread
            )),

            think=os.getenv(
                "OLLAMA_THINK",
                str(defaults.think)
            ).lower() in ("true", "1", "yes"),

            keep_alive=os.getenv(
                "OLLAMA_KEEP_ALIVE",
                defaults.keep_alive
            ),

            connect_timeout=int(os.getenv(
                "OLLAMA_CONNECT_TIMEOUT",
                defaults.connect_timeout
            )),

            read_timeout=int(os.getenv(
                "OLLAMA_READ_TIMEOUT",
                defaults.read_timeout
            )),

            max_history_messages=int(os.getenv(
                "STARMENTAI_MAX_HISTORY",
                defaults.max_history_messages
            )),

            max_user_input_length=int(os.getenv(
                "STARMENTAI_MAX_INPUT",
                defaults.max_user_input_length
            )),
        )


CONFIG: Final[AppConfig] = AppConfig.from_env()
