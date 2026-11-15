
"""
StarmentAI - Validator
Validates and normalizes application input and output.
"""

from config import CONFIG


class ValidationError(Exception):
    """Base exception for validation errors."""


class Validator:
    ALLOWED_ROLES = {
        "system",
        "user",
        "assistant",
    }

    @staticmethod
    def validate_user_input(value: str) -> str:
        """Validate and normalize user input."""

        if not isinstance(value, str):
            raise ValidationError(
                "User input must be a string."
            )

        value = value.strip()

        if not value:
            raise ValidationError(
                "User input cannot be empty."
            )

        if len(value) > CONFIG.max_user_input_length:
            raise ValidationError(
                f"Input exceeds the maximum length of "
                f"{CONFIG.max_user_input_length} characters."
            )

        return value

    @classmethod
    def validate_messages(
        cls,
        messages: list[dict],
    ) -> list[dict[str, str]]:
        """Validate and normalize a message list."""

        if not isinstance(messages, list):
            raise ValidationError(
                "Messages must be a list."
            )

        if not messages:
            raise ValidationError(
                "Messages cannot be empty."
            )

        normalized = []

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(
                    f"Message {index} must be a dictionary."
                )

            role = message.get("role")
            content = message.get("content")

            if role not in cls.ALLOWED_ROLES:
                raise ValidationError(
                    f"Message {index} has an invalid role."
                )

            if not isinstance(content, str):
                raise ValidationError(
                    f"Message {index} content must be a string."
                )

            content = content.strip()

            if not content:
                raise ValidationError(
                    f"Message {index} cannot be empty."
                )

            normalized.append({
                "role": role,
                "content": content,
            })

        return normalized

    @staticmethod
    def validate_response(response: str) -> str:
        """Validate and normalize AI response."""

        if not isinstance(response, str):
            raise ValidationError(
                "AI response must be a string."
            )

        response = response.strip()

        if not response:
            raise ValidationError(
                "AI response cannot be empty."
            )

        return response

    @staticmethod
    def validate_boolean(value) -> bool:
        """Ensure a value is a boolean."""

        if not isinstance(value, bool):
            raise ValidationError(
                "Value must be a boolean."
            )

        return value
