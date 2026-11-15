
import unittest

from core.validator import Validator, ValidationError


class TestValidator(unittest.TestCase):
    def test_valid_user_input(self):
        result = Validator.validate_user_input("Hello")

        self.assertEqual(result, "Hello")

    def test_user_input_is_trimmed(self):
        result = Validator.validate_user_input("  Hello  ")

        self.assertEqual(result, "Hello")

    def test_empty_user_input(self):
        with self.assertRaises(ValidationError):
            Validator.validate_user_input("   ")

    def test_non_string_user_input(self):
        with self.assertRaises(ValidationError):
            Validator.validate_user_input(123)

    def test_user_input_too_long(self):
        from unittest.mock import patch
        from types import SimpleNamespace

        with patch(
            "core.validator.CONFIG",
            SimpleNamespace(max_user_input_length=5),
        ):
            with self.assertRaises(ValidationError):
                Validator.validate_user_input("123456")

    def test_valid_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = Validator.validate_messages(messages)

        self.assertEqual(result, messages)

    def test_messages_are_trimmed(self):
        messages = [
            {"role": "user", "content": "  Hello  "}
        ]

        result = Validator.validate_messages(messages)

        self.assertEqual(
            result[0]["content"],
            "Hello",
        )

    def test_messages_must_be_list(self):
        with self.assertRaises(ValidationError):
            Validator.validate_messages("invalid")

    def test_empty_messages(self):
        with self.assertRaises(ValidationError):
            Validator.validate_messages([])

    def test_invalid_message_type(self):
        with self.assertRaises(ValidationError):
            Validator.validate_messages(["invalid"])

    def test_invalid_role(self):
        with self.assertRaises(ValidationError):
            Validator.validate_messages([
                {"role": "unknown", "content": "Hello"}
            ])

    def test_invalid_message_content(self):
        with self.assertRaises(ValidationError):
            Validator.validate_messages([
                {"role": "user", "content": 123}
            ])

    def test_empty_message_content(self):
        with self.assertRaises(ValidationError):
            Validator.validate_messages([
                {"role": "user", "content": "   "}
            ])

    def test_valid_response(self):
        result = Validator.validate_response("Hello")

        self.assertEqual(result, "Hello")

    def test_response_is_trimmed(self):
        result = Validator.validate_response("  Hello  ")

        self.assertEqual(result, "Hello")

    def test_empty_response(self):
        with self.assertRaises(ValidationError):
            Validator.validate_response(" ")

    def test_non_string_response(self):
        with self.assertRaises(ValidationError):
            Validator.validate_response(123)

    def test_valid_boolean(self):
        self.assertTrue(Validator.validate_boolean(True))
        self.assertFalse(Validator.validate_boolean(False))

    def test_invalid_boolean(self):
        with self.assertRaises(ValidationError):
            Validator.validate_boolean(1)


if __name__ == "__main__":
    unittest.main()
