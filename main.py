
"""
StarmentAI - Main Application
Terminal entry point.

Mặc định chạy giao diện TUI dạng opencode (tui.py).
Dùng --plain để chạy chế độ terminal cũ, --demo để chạy TUI không cần Ollama.
"""

import argparse
import sys

from config import CONFIG
from core.processor import Processor, ProcessorError


APP_NAME = CONFIG.app_name
APP_VERSION = CONFIG.app_version

COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear conversation history",
    "/exit": "Exit StarmentAI",
}


def print_banner() -> None:
    """Display application banner."""
    print("=" * 50)
    print(f"             {APP_NAME}")
    print(f"             Version {APP_VERSION}")
    print("=" * 50)
    print("AI assistant powered by local Ollama.")
    print("Type /help to see available commands.")
    print()


def print_help() -> None:
    """Display available commands."""
    print("\nAvailable commands:")

    for command, description in COMMANDS.items():
        print(f"  {command:<8} {description}")

    print()


def handle_command(command: str, processor: Processor) -> bool:
    """
    Handle terminal commands.

    Returns:
        True if the application should continue.
        False if the application should exit.
    """

    if command == "/exit":
        print(f"\n{APP_NAME}: Goodbye!")
        return False

    if command == "/help":
        print_help()
        return True

    if command == "/clear":
        processor.clear_conversation()
        print("\nConversation history cleared.\n")
        return True

    return True


def write_smoothly(chunk: str) -> None:
    """Print a streamed chunk directly without artificial delay."""
    sys.stdout.write(chunk)
    sys.stdout.flush()


def print_context_warning(processor: Processor) -> None:
    """Warn when the estimated context exceeds the message budget."""
    usage = processor.last_context_usage

    if not usage or not usage.get("exceeds_message_budget"):
        return

    print(
        "\n[WARNING] Context vượt ngân sách: "
        f"{usage['estimated_tokens']} / {usage['message_budget']} token "
        f"(giới hạn {usage['context_limit']}, "
        f"reserved {usage['reserved_tokens']}). "
        "Hội thoại cũ đã bị cắt bớt.\n"
    )


def process_input(user_input: str, processor: Processor) -> None:
    """Validate and process user input."""

    thinking = processor.will_think(user_input)

    print(f"\n{APP_NAME}: ", end="", flush=True)

    if thinking:
        print("(thinking...)")
        print(f"{APP_NAME}: ", end="", flush=True)

    try:
        for chunk in processor.process_stream(user_input):
            write_smoothly(chunk)

    finally:
        print("\n")
        print_context_warning(processor)


def run_plain() -> None:
    """Main application loop (plain terminal mode)."""

    print_banner()

    if not CONFIG.system_prompt:
        print(
            "Warning: system prompt was not loaded from Modelfile.\n"
            "Run the app from the project directory so the "
            "StarmentAI persona is applied.\n"
        )

    try:
        with Processor() as processor:

            while True:
                try:
                    user_input = input("You: ").strip()

                    if not user_input:
                        continue

                    command = user_input.lower()

                    if command.startswith("/"):

                        if command not in COMMANDS:
                            print("\nUnknown command. Type /help for help.\n")
                            continue

                        if not handle_command(command, processor):
                            break

                        continue

                    process_input(user_input, processor)

                except ProcessorError as exc:
                    print(f"\nProcessing error: {exc}\n")

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user.")
                    break

                except EOFError:
                    print("\n\nInput stream closed.")
                    break

    except KeyboardInterrupt:
        print(f"\n{APP_NAME} stopped.")

    except Exception as exc:
        print(f"\nApplication error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Dùng giao diện terminal đơn giản (không TUI)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Chạy TUI dạng opencode với processor giả (không cần Ollama)",
    )
    args = parser.parse_args()

    if args.plain:
        run_plain()
        return

    try:
        from tui import run_tui
    except ImportError:
        print(
            "Thiếu thư viện textual. Cài bằng: pip install textual\n"
            "Hoặc chạy chế độ cũ: python main.py --plain"
        )
        return

    run_tui(demo=args.demo)


if __name__ == "__main__":
    main()