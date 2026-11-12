"""Проверки публичного интерфейса интеграции."""

import unittest
from integration.interface import EnergytoolAI


class IntegrationTests(unittest.TestCase):
    """Проверяет валидацию идентификатора сессии."""

    def test_invalid_session_id_is_rejected(self):
        """Не допускает путь к файлу вместо UUID сессии."""
        with self.assertRaises(ValueError):
            EnergytoolAI._get_session("../../some-file")


if __name__ == "__main__":
    unittest.main()
