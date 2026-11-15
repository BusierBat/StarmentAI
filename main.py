
"""
StarmentAI - Main Application
Terminal entry point.
"""

from core.processor import Processor, ProcessorError
from core.validator import Validator, ValidationError


APP_NAME = "StarmentAI"                                                 
APP_VERSION = "0.1.0"


def print_banner():
    print("=" * 48)
    print(f"             {APP_NAME}")
    print(f"             Version {APP_VERSION}")
    print("=" * 48)
    print("AI assistant powered by local Ollama.")
    print("Type /help to see available commands.")
    print()


def print_help():
    print("\nAvailable commands:")
    print("  /help   Show this help message")
    print("  /clear  Clear conversation history")
    print("  /exit   Exit StarmentAI")
    print()


def print_response(response: str):
    print(f"\nStarmentAI: {response}\n")


def run():
    print_banner()

    try:
        with Processor() as processor:
            while True:
                try:
                    user_input = input("You: ").strip()

                    if not user_input:
                        continue

                    command = user_input.lower()

                    if command == "/exit":
                        print("\nStarmentAI: Goodbye!")
                        break

                    if command == "/help":
                        print_help()
                        continue

                    if command == "/clear":
                        processor.clear_conversation()
                        print("\nConversation history cleared.\n")
                        continue

                    validated_input = (
                        Validator.validate_user_input(user_input)
                    )

                    result = processor.process(
                        validated_input
                    )

                    print_response(result["response"])

                except ValidationError as exc:
                    print(f"\nInput error: {exc}\n")

                except ProcessorError as exc:
                    print(f"\nProcessing error: {exc}\n")

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user.")
                    break

                except EOFError:
                    print("\n\nInput stream closed.")
                    break

    except KeyboardInterrupt:
        print("\nStarmentAI stopped.")

    except Exception as exc:
        print(f"\nApplication error: {exc}")


if __name__ == "__main__":
    run()
