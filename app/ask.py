"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive "LoRAとは何ですか？"
    uv run python -m app.ask --compare-models gemini-3.5-flash-lite gemini-3.5-flash \
        --compare-questions "LoRAとは何ですか？" "QLoRAとの違いは？"

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

from google import genai
from google.genai.chats import Chat
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass
class SessionMetrics:
    """1つのプロセスで発生した API 呼び出しの集計。"""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0

    def record(self, response: types.GenerateContentResponse, elapsed_seconds: float) -> None:
        usage = response.usage_metadata
        self.requests += 1
        self.input_tokens += usage.prompt_token_count or 0
        self.output_tokens += usage.candidates_token_count or 0
        self.elapsed_seconds += elapsed_seconds


@dataclass
class ComparisonResult:
    model: str
    question: str
    response_time: float
    input_tokens: int
    output_tokens: int
    answer: str


def record_usage(
    response: types.GenerateContentResponse,
    elapsed_seconds: float,
    metrics: SessionMetrics,
) -> tuple[int, int]:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    metrics.record(response, elapsed_seconds)
    return input_tokens, output_tokens


def print_usage(
    response: types.GenerateContentResponse,
    elapsed_seconds: float,
    metrics: SessionMetrics,
) -> None:
    input_tokens, output_tokens = record_usage(response, elapsed_seconds, metrics)
    print(
        f"[metrics] response_time={elapsed_seconds:.2f}s, "
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
        f"total_tokens={input_tokens + output_tokens}"
    )
    print(
        f"[session] requests={metrics.requests}, "
        f"input_tokens={metrics.input_tokens}, "
        f"output_tokens={metrics.output_tokens}, "
        f"total_tokens={metrics.input_tokens + metrics.output_tokens}"
    )
    print(
        "[rate-limit] 無料枠はモデル・アカウントごとに RPM（リクエスト数）、"
        "TPM（トークン数）、RPD（1日あたりのリクエスト数）などの上限があります。"
    )
    print(
        "[rate-limit] この実行のリクエスト数とトークン数を表示しています。"
        "正確な残量・上限は Google AI Studio の使用量画面で確認してください。"
    )


def ask_in_chat(
    chat: Chat,
    question: str,
    metrics: SessionMetrics | None = None,
) -> str:
    """同じ Chat に質問を送り、これまでの会話履歴を引き継ぐ。"""
    if metrics is None:
        metrics = SessionMetrics()
    started_at = time.perf_counter()
    response = chat.send_message(question)
    print_usage(response, time.perf_counter() - started_at, metrics)
    return response.text


def compare_models(models: list[str], questions: list[str]) -> list[ComparisonResult]:
    """各モデル・質問の組み合わせを独立したチャットで実行して比較する。"""
    client = genai.Client(api_key=get_api_key())
    results = []
    for model in models:
        for question in questions:
            started_at = time.perf_counter()
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(temperature=1.0),
            )
            response = chat.send_message(question)
            elapsed_seconds = time.perf_counter() - started_at
            input_tokens, output_tokens = record_usage(
                response, elapsed_seconds, SessionMetrics()
            )
            results.append(
                ComparisonResult(
                    model=model,
                    question=question,
                    response_time=elapsed_seconds,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    answer=response.text,
                )
            )
    return results


def print_comparison(results: list[ComparisonResult]) -> None:
    """モデル・質問ごとの計測結果を比較表として表示する。"""
    print("\n[comparison]")
    print("model\tquestion\tresponse_time\tinput_tokens\toutput_tokens\ttotal_tokens")
    for result in results:
        print(
            f"{result.model}\t{result.question}\t{result.response_time:.2f}s\t"
            f"{result.input_tokens}\t{result.output_tokens}\t"
            f"{result.input_tokens + result.output_tokens}"
        )
    print("\n[answers]")
    for result in results:
        print(f"\n[{result.model}] {result.question}\n{result.answer}")
    print(
        "\n[rate-limit] 比較ではモデル数×質問数の回数だけ API を呼びます。"
        "各モデルの RPM/TPM/RPD の無料枠を消費するため、上限に注意してください。"
    )


def ask(question: str) -> str:
    """質問を1回だけ送る（従来の単発実行）。"""
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(temperature=1.0),
    )
    return ask_in_chat(chat, question, SessionMetrics())


def interactive_chat(
    initial_question: str,
    input_fn: Callable[[str], str] = input,
) -> None:
    """終了するまで、同じ Chat に質問を送り続ける。"""
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(temperature=1.0),
    )
    metrics = SessionMetrics()

    question = initial_question
    while True:
        print(f"[質問] {question}\n")
        print(ask_in_chat(chat, question, metrics))
        try:
            question = input_fn("\n次の質問（終了するには quit）: ").strip()
        except EOFError:
            print()
            return
        if question.lower() in {"quit", "exit"}:
            return
        while not question:
            print("質問を入力してください。")
            try:
                question = input_fn("次の質問（終了するには quit）: ").strip()
            except EOFError:
                print()
                return
            if question.lower() in {"quit", "exit"}:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="同じ会話履歴を使って続けて質問する",
    )
    parser.add_argument(
        "--compare-models",
        nargs="+",
        metavar="MODEL",
        help="指定したモデルと質問の全組み合わせを比較する",
    )
    parser.add_argument(
        "--compare-questions",
        nargs="+",
        metavar="QUESTION",
        help="比較する質問（--compare-models と併用）",
    )
    args = parser.parse_args()

    try:
        if bool(args.compare_models) != bool(args.compare_questions):
            parser.error("--compare-models と --compare-questions は併用してください")
        if args.compare_models and args.compare_questions:
            print_comparison(compare_models(args.compare_models, args.compare_questions))
        elif args.interactive:
            print(f"[model] {MODEL}")
            interactive_chat(args.question)
        else:
            print(f"[model] {MODEL}")
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
