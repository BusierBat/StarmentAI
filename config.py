
"""
StarmentAI - Configuration
Central configuration for the application.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
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

    @property
    def chat_url(self) -> str:
        return (
            f"{self.ollama_host.rstrip('/')}"
            f"{self.ollama_chat_endpoint}"
        )


CONFIG = AppConfig()
