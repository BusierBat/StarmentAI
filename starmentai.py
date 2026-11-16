"""Small health-check script for the local StarmentAI model."""

from core.ollama_client import OllamaClient, OllamaError


def main() -> int:
    try:
        with OllamaClient() as client:
            response = client.chat(
                [{"role": "user", "content": "Xin chào!"}],
                think=False,
            )
    except OllamaError as exc:
        print(f"StarmentAI health check failed: {exc}")
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())