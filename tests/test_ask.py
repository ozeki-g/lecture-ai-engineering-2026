from types import SimpleNamespace

from app import ask


class FakeChat:
    def __init__(self):
        self.questions = []

    def send_message(self, question):
        self.questions.append(question)
        return SimpleNamespace(
            text=f"回答: {question}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=2,
            ),
        )


class FakeClient:
    def __init__(self, chat):
        self.chat = chat
        self.chats = self

    def create(self, *, model, config):
        assert model == ask.MODEL
        assert config.temperature == 1.0
        return self.chat


def test_interactive_chat_reuses_the_same_chat(monkeypatch, capsys):
    chat = FakeChat()
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: FakeClient(chat))

    questions = iter(["それは何に使いますか？", "quit"])
    ask.interactive_chat("LoRAとは何ですか？", lambda _: next(questions))

    assert chat.questions == ["LoRAとは何ですか？", "それは何に使いますか？"]
    output = capsys.readouterr().out
    assert "回答: それは何に使いますか？" in output
    assert "[metrics] response_time=" in output
    assert "[session] requests=2" in output
    assert "[rate-limit]" in output


def test_ask_sends_a_single_question(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: FakeClient(chat))

    assert ask.ask("LoRAとは何ですか？") == "回答: LoRAとは何ですか？"
    assert chat.questions == ["LoRAとは何ですか？"]


def test_session_metrics_accumulates_usage():
    metrics = ask.SessionMetrics()
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=3,
            candidates_token_count=4,
        )
    )

    metrics.record(response, 1.25)

    assert metrics.requests == 1
    assert metrics.input_tokens == 3
    assert metrics.output_tokens == 4
    assert metrics.elapsed_seconds == 1.25


def test_compare_models_runs_every_model_question_combination(monkeypatch):
    class ComparisonChat:
        def __init__(self, model):
            self.model = model

        def send_message(self, question):
            return SimpleNamespace(
                text=f"{self.model}: {question}",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=len(question),
                    candidates_token_count=2,
                ),
            )

    class ComparisonClient:
        def __init__(self):
            self.chats = self

        def create(self, *, model, config):
            return ComparisonChat(model)

    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: ComparisonClient())
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    results = ask.compare_models(["model-a", "model-b"], ["質問1", "質問2"])

    assert [(r.model, r.question) for r in results] == [
        ("model-a", "質問1"),
        ("model-a", "質問2"),
        ("model-b", "質問1"),
        ("model-b", "質問2"),
    ]
    assert results[0].input_tokens == len("質問1")
    assert results[0].output_tokens == 2
    assert results[0].response_time >= 0
