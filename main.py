
"""
StarmentAI - Main Application
Terminal entry point.
"""

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


def process_input(user_input: str, processor: Processor) -> None:
    """Validate and process user input."""

    print(f"\n{APP_NAME}: ", end="", flush=True)

    try:
        for chunk in processor.process_stream(user_input):
            print(chunk, end="", flush=True)

    finally:
        print("\n")


def run() -> None:
    """Main application loop."""

    print_banner()

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


if __name__ == "__main__":
    run()
