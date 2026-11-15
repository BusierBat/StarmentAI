
"""
StarmentAI - Configuration
Centralized application configuration.
"""

import os

from dataclasses import dataclass
from typing import Final


def _load_system_prompt(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return ""

    marker = 'SYSTEM """'
    start = text.find(marker)

    if start == -1:
        return ""

    start += len(marker)
    end = text.find('"""', start)

    if end == -1:
        return ""

    return text[start:end].strip()


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
    temperature: float = 0.6
    top_p: float = 0.9
    num_ctx: int = 3072
    num_thread: int = 4
    num_predict: int | None = 1024
    system_prompt_tokens: int = 704
    think: bool = False
    adaptive_think: bool = True

    system_prompt: str = ""
    style_reminder: str = (
        "Nhắc lại quy tắc của StarmentAI: trả lời như chat; "
        "chào hỏi, cảm ơn, toán đơn thì ngắn gọn đúng ý; "
        "câu hỏi có nội dung thì trả lời đủ ý, nêu lý do và "
        "ví dụ cụ thể, tránh một câu cụt; giọng thẳng, sắc, "
        "châm biếm thông minh, tiếng lóng tự nhiên khi hợp cảnh, "
        "không công kích cá nhân; chủ đề nhạy cảm thì nghiêm túc; "
        "trả lời đúng câu hỏi vừa đặt, bám sát hội thoại, "
        "không sáo rỗng, không đổi chủ đề; "
        "dùng đúng ngôn ngữ của người dùng."
    )

    # Model memory management
    keep_alive: str = "5m"

    # Request settings
    connect_timeout: int = 10
    read_timeout: int = 300

    # Conversation limits
    max_history_messages: int = 12
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

        if self.num_predict is not None and self.num_predict <= 0:
            raise ValueError("num_predict must be greater than 0.")

        if self.system_prompt_tokens < 0:
            raise ValueError(
                "System prompt tokens cannot be negative."
            )

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

            num_predict=(
                int(os.environ["OLLAMA_NUM_PREDICT"])
                if os.getenv("OLLAMA_NUM_PREDICT")
                else defaults.num_predict
            ),

            system_prompt_tokens=int(os.getenv(
                "STARMENTAI_SYSTEM_PROMPT_TOKENS",
                defaults.system_prompt_tokens
            )),

            system_prompt=_load_system_prompt(
                os.getenv("STARMENTAI_MODELFILE", "Modelfile")
            ),

            style_reminder=os.getenv(
                "STARMENTAI_STYLE_REMINDER",
                defaults.style_reminder
            ),

            think=os.getenv(
                "OLLAMA_THINK",
                str(defaults.think)
            ).lower() in ("true", "1", "yes"),

            adaptive_think=os.getenv(
                "STARMENTAI_ADAPTIVE_THINK",
                str(defaults.adaptive_think)
            ).lower() in ("true", "1", "yes", "on"),

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

print("")