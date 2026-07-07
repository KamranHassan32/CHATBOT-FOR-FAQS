import types

import app as app_module
from app import find_best_match


def test_returns_best_answer_for_known_question():
    result = find_best_match("How long does shipping take?")
    assert result["matched_question"] == "How long does shipping take?"
    assert result["score"] > 0.2
    assert "business days" in result["answer"].lower()


def test_returns_fallback_for_unknown_question():
    result = find_best_match("Tell me about alien technology")
    assert result["matched_question"] is None
    assert "close match" in result["answer"].lower()


def test_get_chatgpt_reply_returns_openai_answer_when_available(monkeypatch):
    class DummyCompletions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Paris"))]
            )

    class DummyClient:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=DummyCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "client", DummyClient("test-key"))

    assert app_module.get_chatgpt_reply("What is the capital of France?") == "Paris"
