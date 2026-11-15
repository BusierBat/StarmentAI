
"""
StarmentAI - Validator
Validates and normalizes application input and output.
"""

from typing import Any

from config import CONFIG


class ValidationError(Exception):
    """Base exception for validation errors."""


class Validator:
    """Centralized validation utilities."""

    ALLOWED_ROLES = frozenset({
        "system",
        "user",
        "assistant",
    })

    # --------------------------------------------------
    # Internal validation
    # --------------------------------------------------

    @staticmethod
    def _validate_string(
        value: Any,
        field_name: str,
    ) -> str:
        """Validate and normalize a string."""

        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValidationError(
                f"{field_name} cannot be empty."
            )

        return value

    # --------------------------------------------------
    # User input
    # --------------------------------------------------

    @staticmethod
    def validate_user_input(value: str) -> str:
        """Validate and normalize user input."""

        value = Validator._validate_string(
            value,
            "User input",
        )

        limit = CONFIG.max_user_input_length

        if len(value) > limit:
            raise ValidationError(
                f"Input exceeds the maximum length "
                f"of {limit} characters."
            )

        return value

    # --------------------------------------------------
    # Messages
    # --------------------------------------------------

    @classmethod
    def validate_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Validate and normalize conversation messages."""

        if not isinstance(messages, list):
            raise ValidationError(
                "Messages must be a list."
            )

        if not messages:
            raise ValidationError(
                "Messages cannot be empty."
            )

        normalized: list[dict[str, str]] = []

        for index, message in enumerate(messages):

            if not isinstance(message, dict):
                raise ValidationError(
                    f"Message {index} must be a dictionary."
                )

            role = message.get("role")

            if not isinstance(role, str) or role not in cls.ALLOWED_ROLES:
                raise ValidationError(
                    f"Message {index} has an invalid role."
                )

            content = cls._validate_string(
                message.get("content"),
                f"Message {index} content",
            )

            normalized.append({
                "role": role,
                "content": content,
            })

        return normalized

    # --------------------------------------------------
    # AI response
    # --------------------------------------------------

    @staticmethod
    def validate_response(response: str) -> str:
        """Validate and normalize AI response."""

        return Validator._validate_string(
            response,
            "AI response",
        )

    # --------------------------------------------------
    # Boolean
    # --------------------------------------------------

    @staticmethod
    def validate_boolean(value: Any) -> bool:
        """Validate boolean values."""

        if not isinstance(value, bool):
            raise ValidationError(
                "Value must be a boolean."
            )

        return value

    # --------------------------------------------------
    # Numeric
    # --------------------------------------------------

    @staticmethod
    def validate_positive_integer(
        value: Any,
        field_name: str = "Value",
    ) -> int:
        """Validate a positive integer."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(
                f"{field_name} must be an integer."
            )

        if value <= 0:
            raise ValidationError(
                f"{field_name} must be greater than zero."
            )

        return value

