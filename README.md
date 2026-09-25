# voicelab

同じ音声エージェントを **3 通りで作って比べる**ための実験場。「どの構成をいつ選ぶか」を、
数字で言えるようにするのが目的。

- **A: Agents Platform** … ElevenLabs の Agent（音声認識・応答生成・発話・割り込みを丸ごと任せる）。ノートは手元に置き、Agent が道具（client tool）で引く
- **B: 自前構成** … 自前の検索 + Gemini + ストリーミング TTS を自分でつなぐ。ノートは手元に置き、LLM に選ばせず先に引く
- **C: Knowledge Base** … ノートの本文を **ElevenLabs 側にアップロードして預け**、向こうの RAG で引かせる。道具は持たせない

違いは「[3 つの構成](#3-つの構成)」の表にまとめてある。

比べる観点は 3 つ。

| 観点 | 測り方 |
|---|---|
| 遅延 | 質問を投げてから最初の音が出るまで（ms）。自前で計る（[計測の定義](#計測の定義)） |
| 費用 | 1 往復あたりのクレジット。A と C は会話ごとの `cost`（分単位）、B は TTS の文字数（[B の設計](#b-の設計)）。どれも実行前後の残高の差と突き合わせる |
| 実装量 | 行数と、自分で面倒を見る必要があるものの数（割り込み、無音判定、再接続…） |

## 3 つの構成

**何を ElevenLabs に渡し、何を手元に置くか**が構成の正体で、遅延も費用も運用の面倒さも
そこから出てくる。

| | A: Agents Platform | B: 自前構成 | C: Knowledge Base |
|---|---|---|---|
| ノートの本文 | **手元**（`corpus/`） | **手元**（`corpus/`） | **ElevenLabs に預ける**（アップロード） |
| 検索の実装 | **手元**（`search.py` を道具として呼ばれる） | **手元**（`search.py` を先に呼ぶ） | **ElevenLabs 側**（内部の RAG） |
| 検索を呼ぶ判断 | LLM（道具を選ぶ） | こちら（毎回必ず引く） | ElevenLabs 側（`usage_mode: auto`） |
| LLM の呼び出し | **ElevenLabs 側** | **自分**（Gemini API） | **ElevenLabs 側** |
| 音声（TTS） | ElevenLabs 側 | ElevenLabs の TTS に自分でつなぐ | ElevenLabs 側 |
| Agent が持つ道具 | `search_notes` 1 つ | なし（Agent を使わない） | **なし** |
| ツール往復 | **あり**（実測 1.5〜2 秒） | なし | なし |
| ノートを直したとき | そのまま反映される | そのまま反映される | **`setup-kb` で同期が要る** |
| ノートが外に出るか | 出ない | 出ない | **出る** |
| 費用の出方 | 会話の `cost` に全部込み | TTS の文字数 ＋ Gemini のトークン（二社に分かれる） | 会話の `cost` に全部込み |

読み方:

- **A と B はどちらもノートを手元に置いている。** 違うのは「誰が検索を呼ぶか」と
  「LLM を誰が動かすか」だけ
- **C だけが本文を渡している。** そのぶんツール往復が消えるので**一番速い可能性がある**が、
  ノートが外に出て、直すたびに同期が要る。社外に出せないノートでは選べない
- 検索の質は 3 つで**揃わない**。A・B は同じ `search.py`（文字バイグラム）、C は ElevenLabs の
  RAG（`multilingual_e5_large_instruct` の埋め込み）。遅延と費用は比べられるが、
  正誤は「別の検索どうしの比較」になることに注意する

### C の作り方（`setup-kb`）

```powershell
python run.py setup-kb            # corpus/ の 3 本を登録 → 索引 → Agent voicelab-c
python run.py run kb --dry-run --scenario deploy-check
python run.py run kb --scenario deploy-check      # 1 問（課金あり）
```

- `setup-kb` は**名前を鍵にして冪等**。文書名は `search.py` が使うノート名（ファイル名から
  拡張子を落としたもの）と揃えてある。既にあれば作り直さず使い回し、索引を確かめて
  Agent を更新する。`corpus/` の**本文そのもの**を直したときだけ `--recreate` を付ける
  （確認済みの API に本文を差し替える口が無く、消してから入れ直すしかない）
- 索引の埋め込みは **`multilingual_e5_large_instruct` で固定**。既定の
  `e5_mistral_7b_instruct` は英語寄りで、日本語のノートでは引けない
- 索引は**出来上がるまで待つ**（2 秒間隔、最大 120 秒）。状態は `created` →
  `processing` → `succeeded` と進む。**`created` はまだ引けない**ので、
  `succeeded` だけを出来上がりとみなしている。初回は 1 本あたり 5〜11 秒だった
- 作った Agent の id は `.env` の `ELEVENLABS_KB_AGENT_ID` に書く（A の
  `ELEVENLABS_AGENT_ID` とは別。両方を残して測り比べるため）
- 声・TTS モデル・LLM・`turn`・`client_events`・`max_duration_seconds`・
  `text_normalisation_type`・認証は **A と同じ値**にしてある
  （`agent_setup.build_agent_body` を A と C で共有）。prompt も A と同じ文言で、
  違うのは冒頭の 2 行（「必ず `search_notes` を呼ぶ」→「ノートを参照できます」）だけ
- `run kb --dry-run` は会話せずに「預けてある文書が何本か」「RAG の埋め込みが日本語向きか」
  「道具が空か」を表示する（クレジットを使わない）

### C が RAG を使ったことをどう確かめるか

会話メタデータ（`GET /v1/convai/conversations/{id}`）の `metadata.rag_usage` を
書き起こし（`results/transcripts/<id>_kb_<UTC>.txt`）に丸ごと残している。
これが空なら「速かったのは何も引かなかったから」を疑う。

```
## RAG（ElevenLabs 側の検索）
- rag_usage: {"usage_count": 1, "embedding_model": "multilingual_e5_large_instruct"}
```

## なぜ web app ではないのか

共有できる形の音声対話は `obsidian-vault-web` の `/voice`（ブラウザ + WebRTC）で既に動いている。
ここで欲しいのは触れるデモではなく**数字**なので、ブラウザの音声処理層を挟まない Python にしている。

## 状態

**A・B とも 5 問流して並べ、正誤も入れた（2026-09-11）。C は 1 問で疎通を確かめたところ（5 問は未実行）。**

- [x] 題材データ（`corpus/`）と質問集（`scenarios/questions.json`）
- [x] クレジット残高の記録（`voicelab/credits.py`）
- [x] 計測結果の記録と表の出力（`voicelab/metrics.py`）
- [x] ノートの検索（`voicelab/search.py`）と Agent 用の client tool
- [x] Agent とツールの作成・更新（`voicelab/agent_setup.py`）
- [x] A: Agents Platform で 1 往復する（`voicelab/agents_path.py`）… 5 問 + 再測 1 回、計 6 往復
- [x] B: 検索 + Gemini + ストリーミング TTS で 1 往復する（`voicelab/custom_path.py`）… **実行待ち**
- [x] 5 問 × 2 構成を流して表にする（下の「A と B を並べる」）
- [x] ノートを Knowledge Base に預けて索引を張る（`voicelab/kb_setup.py`）
- [x] C: Knowledge Base で 1 往復する（`voicelab/kb_path.py`）… **疎通の 1 問だけ。5 問は実行待ち**
- [ ] 5 問 × 3 構成で表を作り直す

## 使い方

### 最初の 1 回だけ（環境を作る）

```powershell
uv venv                       # .venv を作る（python -m venv .venv でも同じ）
uv sync --extra dev           # 依存 + テスト用を入れる
copy .env.example .env        # ELEVENLABS_API_KEY と GEMINI_API_KEY を書く
```

Agent の id は手で書かなくてよい。`setup-agent` が `ELEVENLABS_AGENT_ID`（A）を、
`setup-kb` が `ELEVENLABS_KB_AGENT_ID`（C）を、それぞれ `.env` の該当行だけ書き換える。

### 2 回目以降（uv を使わない）

`.venv` は**ごく普通の venv** で、uv 固有のものは入っていない。一度有効化すれば、
あとは素の Python のコマンドだけで完結する。

```powershell
.venv\Scripts\activate       # 有効化（プロンプトの頭に (elevenlabs) が付く）

python run.py credits                       # 残高
python run.py scenarios                     # 質問集
python run.py run agents --scenario rollback        # A で 1 問（課金あり）
python run.py run custom --scenario rollback        # B で 1 問（課金あり）
python run.py run kb --scenario rollback            # C で 1 問（課金あり）
python run.py run agents --dry-run                  # 接続せず確認だけ（課金なし）
python run.py setup-kb                              # C の登録・索引・Agent（課金なし）
python run.py report                        # 表を作る
pytest -q                                   # テスト
```

`deactivate` で抜ける。

### search.py は何をしているのか（ElevenLabs のコードが 1 行も無い理由）

`voicelab/search.py` は `corpus/` の 3 本を文字バイグラムで引くだけの関数で、
ElevenLabs にも Gemini にも依存していない。それが**そのまま両方の統合点**になっている。

- **A**: ElevenLabs の Agent に `search_notes` という **client tool** を登録してある。
  Agent が「ノートを引く」と判断すると、その呼び出しが会話の WebSocket でこちらに届き、
  `search.search_notes()` が手元で走って戻り値が LLM に渡る（`agents_path.py` の handler）。
  ツールがサーバ側実行（webhook）ではないので、公開 URL もトンネルも要らない
- **B**: LLM に選ばせず、質問が来たら**先に**同じ関数を呼んで、結果を prompt に添える
- **C**: この関数を**使わない**。ノートの本文を ElevenLabs に預けてあり、向こうの RAG が引く。
  `search.py` が残っているのは、A・B と同じ質問で「手元ならどう引けたか」を
  クレジットを使わずに見比べるため（`voicelab run kb --dry-run` は手元の corpus の本数も出す）

**A と B が同じ検索を使うことが、この比較の前提**。検索が違えば、A と B の遅延差が
「音声の経路の違い」なのか「検索の違い」なのか分からなくなる。
また、知識源が無ければ比較そのものが成り立たない（ツール呼び出しの往復＝A の遅延の
1.5〜2 秒ぶんが測れず、「渡した文を読み上げるだけ」の比較になる）。

単独で叩けるようにしてあるのは診断のため。答えがおかしいとき、
**検索が悪いのか LLM が悪いのか**をクレジットを使わずに切り分けられる。

```powershell
python -m voicelab.search ロールバック   # 1 位が期待どおりなら、悪いのは LLM 側
```

### モジュールを直接叩く

CLI のサブコマンドを覚えなくても、**モジュール 1 つを名指しで**動かせる。
どれも `if __name__ == '__main__':` が数行あるだけで、中身は下に書いた関数を呼ぶだけ。

| コマンド | 呼ばれる関数 | 課金 |
|---|---|---|
| `python -m voicelab.credits` | `credits.read_subscription()` | なし |
| `python -m voicelab.search ロールバック` | `search.search()` | なし |
| `python -m voicelab.metrics` | `metrics.render_report(load_runs())` | なし |
| `python -m voicelab.agent_setup` | `agent_setup.setup()` | なし |
| `python -m voicelab.kb_setup` | `kb_setup.setup()` | なし |
| `python -m voicelab.agents_path rollback` | `agents_path.run_scenario()` | **あり** |
| `python -m voicelab.custom_path rollback` | `custom_path.run_scenario()` | **あり** |
| `python -m voicelab.kb_path rollback` | `kb_path.run_scenario()` | **あり** |

下 3 つは 1 往復ぶん課金される。**残高の見張りは `cli.py` にあるので、この叩き方では効かない。**

REPL からでも同じ。CLI を通さずに関数を直接呼べるよう、表示と引数解析は `cli.py` に、
処理は各モジュールに分けてある（テスト 73 件も CLI を通さず関数を直接呼んでいる）。

```python
from voicelab import credits, search
from voicelab.config import load_env

search.search('ロールバック')                                  # 課金なし
credits.read_subscription(load_env()['ELEVENLABS_API_KEY'])  # 鍵を貼らずに済む形
```

### 同じことをする 4 つの書き方

どれも中身は同じ（`pyproject.toml` の `[project.scripts]` が
`voicelab = "voicelab.cli:main"` を宣言しているだけ）。

| 書き方 | 有効化 | 備考 |
|---|---|---|
| `python run.py credits` | 要 | 素の Python らしい形。`run.py` は 3 行 |
| `python -m voicelab credits` | 要 | `voicelab/__main__.py` を通る |
| `voicelab credits` | 要 | インストール時に作られた `.venv\Scripts\voicelab.exe` |
| `uv run voicelab credits` | 不要 | uv が `.venv` を選んでから実行する |

`python voicelab/cli.py` だけは**動かない**。ファイルを直接指定するとパッケージの一部
として読まれず、中の `from . import ...` が解決できないため。`run.py` はそれを避けるために置いてある。

依存を足すときだけ uv（または有効化した状態で `pip install`）が要る。
`uv add <パッケージ>` は `pyproject.toml` と `uv.lock` も更新するので、そちらが本筋。

### uv と pyproject.toml の関係

`pyproject.toml` は **Python 標準の設定ファイル**（PEP 621）で、uv 固有のものではない。
pip も setuptools も同じものを読む。uv は「そこに書いてあるとおりに `.venv` を揃える道具」。

| 場所 | 何を書くか | 誰が読むか |
|---|---|---|
| `[project] dependencies` | 必須の依存 | uv / pip |
| `[project.optional-dependencies]` | 任意の依存（ここでは `dev` と `conversation`） | `uv sync --extra dev` |
| `[project.scripts]` | コマンド名 → 関数（`voicelab = "voicelab.cli:main"`） | インストール時に `.exe` を作る |
| `[build-system]` | パッケージを組む道具（setuptools） | ビルド時 |
| `uv.lock` | 依存の**正確な版**。uv 固有（`package-lock.json` に相当） | uv |

uv のコマンドが実際にやること:

- `uv venv` … `.venv` を作る。名前を省くと `.venv`（ドット付き）。`python -m venv .venv` と同じ
- `uv sync` … `pyproject.toml` と `uv.lock` のとおりに `.venv` を**揃える**。
  足りないものを入れ、**余計なものを消す**。`--extra dev` を付け忘れると pytest が消えるのはこのため
- `uv add <パッケージ>` … `pyproject.toml` に 1 行足し、`uv.lock` を更新し、`.venv` に入れる。
  有効化した状態の `pip install` でも `.venv` には入るが、`pyproject.toml` は更新されない
- `uv run <コマンド>` … `.venv` を選んでから実行する。有効化していれば要らない

つまり **uv が要るのは環境を作るときと依存を足すときだけ**で、それ以外は素の Python でよい。


## 計測の定義

| 名前 | 意味 |
|---|---|
| `first_audio_ms` | 質問を送ってから、**最初の音の断片**が届くまで。体感の遅延はこれ |
| `reply_done_ms` | 質問を送ってから、**最後の音の断片**が届くまで |
| `credits` | 会話の `metadata.cost`。取れなければ 0 にして `note` に「費用未取得」と書く |

- 質問は**テキストで送る**（`send_user_message`）。マイクを使うと、部屋の雑音と無音判定の
  ばらつきがそのまま数字に乗る。代わりに、**この数字に音声認識の時間は含まれない**。
  B と比べるときは、B も同じくテキスト投入から測ること
- 「言い終わり」は、最初の音が来てから **1.5 秒**音が途切れたら、と決めている。サーバの終了イベント
  ではなく音の途切れで決めるのは、B でも同じ判定が書けるから。判定が違うと数字を並べられない
- 最初の音が **40 秒**来なければ諦める。記録は残し、`note` にタイムアウトと書く
- SDK の `callback_latency_measurement` は**使っていない**。あれは WebSocket の ping（ms）で、
  会話の遅延ではない

## 無音を送る必要があるか

**要らなかった。** マイクを開かず、音声を 1 バイトも送らずに `send_user_message` でテキストを
送るだけで、Agent はツールを呼んで音声で答えた（5 問すべて）。`run_scenario(..., send_silence=True)`
は残してあるが、既定は「送らない」のままでよい。

## なぜトンネルが要らないのか

検索ツールは **client tool**（Python 側で実行）にしている。Agent がツールを呼ぶと、その呼び出しは
会話の WebSocket 経由でこちらに届き、手元の関数の戻り値がそのまま LLM に渡る。ElevenLabs から
HTTP でこちらへ届く経路が無いので、公開 URL も cloudflared のようなトンネルも要らない。
（サーバ側で実行する webhook tool を使う構成なら、ローカル検証にはトンネルが要る。）

## B の設計

> LLM は当初 `gemini-2.5-flash` だったが、2026-09-11 に Gemini API が「新規ユーザーには提供終了」（404）を返したため、A・B とも `gemini-3.6-flash` に揃えた。A の最初の結果（上の表）は 2.5-flash のもので、3.6-flash で取り直した表は下にある。


`voicelab/custom_path.py`。直列のパイプラインで、A と同じ `Run` を返し、WAV と書き起こしも
同じ場所に置く（`results/audio/<id>_custom_<UTC>.wav`、`results/transcripts/<id>_custom_<UTC>.txt`）。

```
質問テキスト → 検索（上位 3 件） → Gemini をストリーミング → 文が確定するたびに TTS へ → 音
```

### A との違い

| | A: Agents Platform | B: 自前構成 |
|---|---|---|
| 検索の呼び方 | **LLM がツールを選ぶ**（client tool） | **選ばせない。先に検索して結果を渡す** |
| ノートに無い質問 | LLM がツールを呼ばずに「見当たりません」 | それでも検索する。0 件を渡して LLM に言わせる |
| LLM | `gemini-3.6-flash`（ElevenLabs 側が動かす） | 同じ `gemini-3.6-flash`（**自分で呼ぶ**） |
| 費用の出方 | 会話の `metadata.cost` に全部込み（分単位） | ElevenLabs は TTS の**文字数**、Gemini は**トークン**。別勘定 |
| 面倒を見るもの | ほぼ無し | 文の切り出し・WebSocket・受信スレッド・時刻の記録 |

ツール往復（LLM がツールを選ぶ → 検索 → LLM が答えを作る）が 1 回消えるぶん、B の方が
速いはず、という仮説を確かめるための構成。A の実測ではその往復が 1.5〜2 秒だった。

### 文単位のストリーミングである

**LLM の生成完了は待たない。** トークンを溜めて「。」「！」「？」で切り、文が 1 つ確定した
時点で TTS へ送る。半角の `.` では切らない（`v2.5` のような版番号で途中まで送ってしまうため）。

TTS は `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input` に
`websockets` で**直接**つなぐ。SDK の `client.text_to_speech.convert_realtime` を使わなかったのは、

- 中の `text_chunker` の区切り文字が `. , ? ! ; : - ( ) [ ] }` と半角空白だけで、**日本語の「。」を知らない**
- その `text_chunker` は「次の断片が来て初めて 1 つ前を送る」作りで、**最初の文が 1 文ぶん遅れる**
- 戻り値が generator なので、音声の到着時刻が「こちらが `next()` を呼んだ時刻」になってしまう

の 3 点。受信は別スレッドで回して、**チャンクが届いた瞬間に時刻を打つ**（A の `MeasuringAudioInterface`
と同じ）。各文は `{"text": "文 ", "flush": true}` で送る。`flush` が無いと `chunk_length_schedule`
（既定 50 文字）ぶん溜まるまで生成が始まらず、短い返答では最初の音が遅れる。
出力は `pcm_16000`。A の録音と同じ 16kHz / 16bit / mono。

Gemini は `thinking_budget=0`（思考なし）で呼ぶ。2〜3 文の読み上げに推論は要らず、
`max_output_tokens` が 300 しかないので、思考でこの枠を使い切って**本文が空のまま終わる**のを避ける。

### 計測

`first_audio_ms` と `reply_done_ms` の定義は A と同じ（[計測の定義](#計測の定義)）。`t0` は
質問テキストを渡した時刻で、**TTS の WebSocket 接続と Gemini の client 作成は `t0` より前**に済ませる
（A も接続後に質問を送っているため）。CSV に入らない内訳は書き起こしに書く。

- 検索 ms / LLM 最初のトークン ms / LLM 完了 ms / TTS へ最初の文 ms
- TTS に送った文の数と文字数
- Gemini の `usage_metadata`（入力・出力・思考・合計トークン）

### 費用

`credits` 列には **TTS に送った文字数 × 0.5** を入れる（flash / turbo 系を API から使ったときの
1 文字あたりのクレジット。根拠と出典は `voicelab/custom_path.py` の `CREDITS_PER_CHARACTER`）。
これは**見積り**で、正は実行前後の残高の差。ただし残高の反映は遅れる。

**Gemini の費用は ElevenLabs のクレジットではないので `credits` に混ぜない。** トークン数として
書き起こしに残す。A ではこの分が会話の `cost` に溶けていて分けられない ―― そこも B との違い。

## 最初の結果（A、2026-09-11）

`results/report.md` と `results/runs.csv` に全行がある。音声は `results/audio/`（git には入れない）。

| 質問 | 最初の音まで | 言い終わりまで | クレジット | ツール |
|---|---:|---:|---:|---|
| デプロイ前の確認 | 2,722 / 2,926 ms | 3,504 / 3,622 ms | 74 / 65 | 呼んだ。1 位が期待どおり |
| ロールバック | 3,252 ms | 3,779 ms | 77 | 同上 |
| RRF とは | 2,987 ms | 3,710 ms | 71 | 同上 |
| 5/12 の定例 | 3,252 ms | 3,646 ms | 70 | 同上 |
| 来週の天気（ノートに無い） | 1,266 ms | 1,374 ms | 44 | 呼ばずに「見当たらない」 |

- **ツールを呼ぶ質問は最初の音まで 2.7〜3.3 秒**、呼ばない質問は 1.3 秒。差の 1.5〜2 秒がツール往復（LLM がツールを選ぶ → 検索 → LLM が答えを作る）のぶん
- 「言い終わり」は最後の音声チャンクが**届いた**時刻。音声はほぼ一括で届くので、再生し終わるのはこの後（録音は 1 問 17 秒前後）
- 費用は **1 往復 44〜77 クレジット**（会話の `metadata.cost`）。通話は 6 秒前後で、分単位の目安（730/分）より安く済んでいるのは、テキスト送信で無音の待ち時間が無いため。長い会話や放置ではその目安に戻る
- 会話メタデータの `cost` 合計は 401 だったが、直後に読んだ残高の差は 216。**残高の反映は遅れる**ので、費用は会話ごとの `cost` を正とする
- 返答は全問ノートを根拠にしていて、出典のノート名を口頭で添えている（`results/transcripts/`）。`correct` 列は**書き起こしの本文**で判定した（期待ノート名を出典として言っているか。無い質問は「見当たらない」と言っているか）。音声の長さと文字数の比（6〜8 字/秒）も全行で妥当で、途中で切れた録音は無い。人が聞いて確かめる価値があるのは読み方だけで、A の「レシピプロカル」（LLM の書いたカタカナが誤り）と「二千二十六年」（漢数字読み）が候補

## A と B を並べる（どちらも gemini-3.6-flash、2026-09-11）

同じ 5 問を、同じ検索関数・同じ声・同じ TTS モデル・同じ LLM で流した。全行は `results/runs.csv`。
**C の列は 5 問を流してから書く（下の「C の疎通確認」を参照）。**

| 質問 | A 最初の音 | B 最初の音 | C 最初の音 | B の内訳: LLM 初トークン | A クレジット | B クレジット（TTS 見積り） | C クレジット |
|---|---:|---:|---:|---:|---:|---:|---:|
| デプロイ前の確認 | 2,479 ms | 3,460 ms | 実行待ち | 3,133 ms | 100 | 55 | 実行待ち |
| ロールバック | 2,653 ms | 6,224 ms | 実行待ち | 5,925 ms | 100 | 50 | 実行待ち |
| RRF とは | 2,633 ms | 4,229 ms | 実行待ち | 3,869 ms | 98 | 50 | 実行待ち |
| 5/12 の定例 | 1,871 ms | 2,845 ms | 実行待ち | 2,397 ms | 88 | 50 | 実行待ち |
| 来週の天気（無い） | 2,049 ms | 1,674 ms | 実行待ち | 1,481 ms | 85 | 16 | 実行待ち |

読み方:

- **A の方が速く、ばらつきも小さい**（1.9〜2.7 秒）。B は 1.7〜6.2 秒で、ほぼ全部が **Gemini の最初のトークンまでの時間**。検索は 1〜2 ms、TTS は最初の文を送ってから 200〜300 ms で音が来る。つまり B の遅さは自前の作りではなく、公開 API 経由の Gemini（`thinking_level='low'` でも思考に 300 トークン前後使う）の応答速度
- A は ElevenLabs の中で同じモデルを呼んでいるはずだが、ツール往復（LLM → ツール → LLM）込みで B の単発呼び出しより速い。中で何をしているか（思考の抑え方、プロビジョニング、地理）はこちらからは見えない。**「同じモデル名でも、誰がどう呼ぶかで 1〜3 秒変わる」**が、この比較で一番大きい発見
- 費用の出方が違う。A は会話ごとに **85〜100 クレジット**（LLM 込み・通話時間ベース）。B は **TTS の文字数ぶん 16〜55 クレジット**＋Gemini のトークン（入力 450 前後 / 出力 60 前後 / 思考 300 前後。ElevenLabs のクレジットではない）。合算すると B の方が安いが、二社に請求が分かれる
- 2.5-flash のときの A（上の表）は 2.7〜3.3 秒だったので、**A は 3.6-flash で 0.5〜1 秒速くなった**。ノートに無い質問は 2.5 ではツールを呼ばず 1.3 秒、3.6 では律儀にツールを呼んで 2.0 秒
- `results/report.md` の A の中央値は 2.5-flash と 3.6-flash の両方の行を含む（`runs.csv` に LLM の列が無いため）。分けて見るときは日時で切る

どちらを選ぶか（この題材での結論。一般化はしない）:

- **速さと手離れ**なら A。割り込み・無音判定・再接続を自分で書かなくてよく、それでいて速い
- **費用の内訳の見える化と、LLM や検索の差し替え自由度**なら B。ただし LLM の応答速度がそのまま体感に出るので、モデルと呼び方（思考の抑制、リージョン）を自分で詰める覚悟が要る

## C の疎通確認（1 問だけ、2026-09-11）

5 問はまだ流していない。**つながることと RAG が引かれることを確かめるために 1 問だけ**流した。

| 質問 | 最初の音 | 言い終わり | クレジット | 通話秒数 | rag_usage |
|---|---:|---:|---:|---:|---|
| デプロイ前の確認 | **1,039 ms** | 1,841 ms | 53 | 4 秒 | `usage_count: 1` / `multilingual_e5_large_instruct` |

- 同じ質問の **A は 2,479 ms・100 クレジット**だったので、**C は 1.4 秒速く、費用は約半分**。
  A の中央値（2,061 ms）と比べても半分に近い。ツール往復（LLM がツールを選ぶ → 検索 →
  LLM が答えを作る）が丸ごと消えたぶんと見て辻褄が合う
- 返答は `デプロイ手順によると、デプロイ前に確認することは4つあります。…` で、
  ノートを根拠にし、出典のノート名を口頭で添え、数字も算用数字だった（A と同じ prompt の効き方）
- 通話 4 秒で 53 クレジット。**A の 85〜100 より安いのは、速く終わったぶん**
  （会話は分単位の課金なので、往復が短いほど安い）
- ただし **1 問では中央値もばらつきも言えない**。5 問流してから表を埋めること

## 声と数字の読み（2026-09-11）

- 当初の声 Sarah は ElevenLabs の検証済み言語に日本語が無い。無料でも使える premade のうち **George / Alice / Jessica は日本語が検証済み**（`GET /v1/voices` の `verified_languages`）。同じ文を 4 声で読ませて聞き比べ（`results/audio/voice-sample_*.wav`）、**Jessica** に切り替えた（`.env` の `ELEVENLABS_VOICE_ID`）
- 年号の読みが不自然だったので、prompt に「算用数字で書く、日付は月日、年は省く」を足した。**B は従うが A は従わない**: A の返答は「五月十二日」「二十件から十件」と漢数字のまま（B は「5月12日」「20から10」）。同じモデル名でも、ElevenLabs の Agent 側で読み上げ向けの整形（数字の漢字化）が挟まっているように見える。TTS がどの書き方を一番自然に読むかは `results/audio/date-sample_*.wav`（Jessica、5 通り）で聞き比べる
- **原因は Agent の TTS 設定 `text_normalisation_type`** だった。既定の `system_prompt` は LLM に「数字を語で書け」と指示するので、日本語では漢数字（五月十二日、二十件）になり、TTS の読みが崩れる。`elevenlabs`（生成後に ElevenLabs 側で整形）にすると LLM は算用数字のまま書き（5月12日、20から10）、読みが直った。遅延の増加は測定誤差の範囲（2.06 秒）。`agent_setup.py` で固定
- B の「たどたどしさ」は文ごとに TTS へ送る作りに由来する可能性がある。同じ文を flash_v2_5 と multilingual_v2 で一括合成した比較が `results/audio/model-sample_*.wav`
- 2026-09-11 16:30 UTC 以降の `runs.csv` の行は Jessica。それ以前は Sarah。18:00 UTC 以降の A は正規化 `elevenlabs`

## 日本語で使う人へ（つまずいた点と対処）

この repo を見て ElevenLabs を日本語で使おうとしている人へ。
**実測して分かったことは `FINDINGS_JA.md` にまとめた。** 要点だけ先に書く。

| つまずき | 対処 |
|---|---|
| 漢字を読み違える（`応答`→えいたい、`通話`→どうが） | **モデルを `eleven_v3` 系にする。** `eleven_multilingual_v2` は日本語で最下位だった |
| Agent が漢数字で喋る（`二千二十六年`） | prompt では直らない。`tts.text_normalisation_type` を `elevenlabs` にする |
| 発音辞書で直そうとしたが効かない | 日本語では `alias` 方式が機能しない。先にモデルを替える |
| 日本語ネイティブの声にしても読み違う | 声ではなくモデルの問題。4 つの声で同じ語が崩れた |
| クローンした声が歪む | 素材の**原音**のピークが 0 dB を超えていないか測る。`loudnorm` を掛けない |
| 検索の埋め込みが日本語で当たらない | 知識ベースの索引は `multilingual_e5_large_instruct` を明示する |

同じ原稿・同じ声でモデルだけ替えた実測（書き起こしと原稿の一致率）:

| モデル | 一致率 |
|---|---:|
| `eleven_v3` / `eleven_v3_conversational` | **100.0%** |
| `eleven_turbo_v2_5` | 99.0% |
| `eleven_flash_v2_5` | 98.5% |
| `eleven_multilingual_v2` | 95.2% |

会話（Agents）でも `eleven_v3_conversational` が選べる。
`eleven_flash_v2_5` から替えても**遅延は悪化せず、1 往復の費用はむしろ下がった**（88〜100 → 59〜64 クレジット）。

## 費用の注意

ここは **A と C（会話）の話**。B は会話ではなく TTS なので、消費は喋った文字数ぶん（1 往復 50 クレジット前後）で、
放置しても増えない。とはいえ残高の見張り（`MIN_CREDITS`）は 3 構成で共通の門をくぐらせている。
**C も会話なので、放置すれば A と同じように分単位で溶ける。**

**会話は分単位でクレジットを消費する。** 2026-09-10 の実測で約 730 クレジット/分だった。
無料プランの 10,000 クレジット/月は実質 13 分、Starter の 30,000 でも 40 分ほどしかない。

- 計測は 1 回で終わらせる。同じシナリオを何度も流さない
- 会話を始めたら必ず終わらせる。放置したタブが 5 分走って 4,168 クレジット（Starter の 14%）消えた事故がある
- `voicelab credits` を実行の前後で回し、`results/runs.csv` に消費を残す

## 題材データについて

`corpus/` は**この実験のために書いた架空のノート**。実際の vault は使わない。

- 個人情報が混ざらないので、そのまま公開できる
- 誰が動かしても同じ結果になるので、比較として意味がある
