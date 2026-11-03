# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：\
素のLLMでは、固有の質問（講義内容など）に対して一般論や「講義資料を確認してください」といったような、一般的な知識に基づいた回答を生成する。\
一方でエージェントでは必要に応じたツールを用いることで、用語集を検索するなど、より固有の用途に特化した機能を実装できる。
- `--trace` で見えたツール呼び出し（ツール名と引数）：\
[tool call]   search_glossary({'query': 'QLoRA'})\
[tool return] search_glossary -> [{"term": "QLoRA", "lecture": 2, "definition": "ベースモデルの重みを4ビットなどに量子化してメモリを節約したうえで、LoRAで学習する手法。より大きなモデルを小さなGPUで扱える。本講座の第2…
- 1つの質問での LLM 呼び出し回数（`requests`）：2回
- 追加したツールと、docstring に書いた説明：\
def make_quiz(lecture: int) -> str:\
    """指定した回の講義内容に基づいたクイズを作る。\
    指定した回の用語から1つを選択し、問題文と正解のペアのリストを返す。\
    「第3回の確認問題を出して」のように、問題の作成を求められた際に使う。\
    """

## 宿題

- 取り組んだレベル：実装 
- やったこと：学習ログを保存する、save_learning_logの実装
- 結果：\
保存に際してユーザーの承認を得る機能を実装し、human in the loopの構造を作ることができた。\
def save_learning_log(log: dict) -> str:\
    """学習ログを保存する。\
    学習の進捗や、モデルがどのような質問をしたかなどを記録するために使う。\
    学習の記録をlearning_log/フォルダにJSON形式で保存する。\
    保存するかどうかは、毎回利用者の確認を挟み、了承が得られた場合にのみ保存する。\
    """\
    LEARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)\
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")\
    path = LEARNING_LOG_DIR / f"{timestamp}.json"\
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")\
    return f"学習ログを保存しました: {path}"

- 気づき・疑問：ユースケースに応じてツールを足していけば実用的なエージェント環境を構築できるかもしれない。
