# voicelab

日本語で話す音声エージェント（声で質問して、手元のノートを根拠に声で答える）を、
**どの社の部品で、どこまで任せて作ると、どれだけ速く・安くなるか**を実測で比べる実験場。
同じ質問 5 問・同じ題材・同じ LLM（Gemini）で流し、違いを 1 か所に絞って数字を並べる。

問いは 2 つ。

1. **音声合成（TTS）をどの社にするか** … 検索と LLM を固定した自前のパイプライン（構成 B）で、
   TTS だけを **ElevenLabs / Deepgram / OpenAI / Google** の 4 社で差し替える。
   検索・prompt・Gemini の呼び方・時刻の打ち方は同じ関数を通るので、差は TTS だけになる
2. **どこまでプラットフォームに任せるか** … 音声認識・応答生成・発話を丸ごと任せる構成（A）、
   ノートまで預ける構成（C）と、自分でつなぐ B を比べる。任せる先は今のところ
   ElevenLabs Agents Platform だけで測っている（他社の同種のサービスはまだ）

比べる観点は 3 つ。

| 観点 | 測り方 |
|---|---|
| 遅延 | 質問を投げてから最初の音が出るまで（ms）。自前で計る（[計測の定義](#計測の定義)） |
| 費用 | 1 往復あたりのドル（各社の定価から見積もり、`usd` 列）。ElevenLabs の会話課金はクレジットでも残す |
| 実装量 | 行数と、自分で面倒を見る必要があるものの数（割り込み、無音判定、再接続…） |

数字の比較は `COMPARISON.md`、各社の口の違いと ElevenLabs の作りは `ARCHITECTURE.md`、
日本語の声で効いたことは `FINDINGS_JA.md`。

書き起こしの「構成:」の行は記号で、**B = ElevenLabs、D = Deepgram、E = OpenAI、F = Google**
（以前の「D: Deepgram 構成」は、今の書き方では B（Deepgram））。

## TTS 4 社の比較（構成 B）

B の TTS だけを差し替える。1 往復の流し方・時刻の打ち方・書き起こしの並びは全社で同じ関数を通るので、
片方だけ直って数字が並べられなくなる事故が起きない。依存は増やしていない（各社の SDK は入れず、
ElevenLabs は `websockets`、他社は既存の `httpx` で叩く）。

| | ElevenLabs | Deepgram | OpenAI | Google |
|---|---|---|---|---|
| モジュール | `custom_path.py` | `deepgram_path.py` | `openai_path.py` | `google_path.py` |
| 口 | `stream-input`（WebSocket。1 本の接続に文を流し込む） | `POST /v1/speak`（REST） | `POST /v1/audio/speech`（REST） | `POST /v1/text:synthesize`（REST） |
| 1 文ごと | 同じ接続に送る | 1 文 1 リクエスト、応答をストリーミングで読む | 1 文 1 リクエスト、chunked で読む | 1 文 1 リクエスト、**1 文ぶんをまとめて**受ける（ストリーミングは gRPC だけ） |
| 既定のモデル / 声 | `.env` の `VOICELAB_TTS_MODEL`（空なら `ELEVENLABS_MODEL_ID`）/ Jessica | `aura-2-izanami-ja`（声とモデルが 1 つの id） | `gpt-4o-mini-tts` / `marin` | `ja-JP-Chirp3-HD-Kore`（声の名前がモデルを兼ねる） |
| 日本語の声 | 検証済みの premade と共有音声（`FINDINGS_JA.md`） | Aura-2 の 5 声 | 組み込み 13 声（日本語専用は無い） | Chirp 3 HD の 30 声 |
| 出力 | `pcm_16000` | `linear16` / 16kHz / `container=none` | `pcm` / **24kHz**（変換せず残す） | `LINEAR16` / 16kHz（先頭 44 バイトの WAV ヘッダを落とす） |
| 課金 | 文字数（クレジット。Flash / Turbo は $0.05 / 1,000 字相当） | 文字数（Aura-2 PAYG $0.030 / 1,000 字） | **トークン**（音 1 分 $0.015 の推定で換算） | 文字数（Chirp 3 HD $0.030 / 1,000 字、**月 100 万字まで無料**） |
| 残高（`credits`） | 読める。`MIN_CREDITS` 未満なら止まる | `billing:read` があれば読める。止まらない。402 なら記録せずに止まる | API では読めない（usage ダッシュボード） | API では読めない（Cloud Console） |
| 書き起こしの記号 | B | D | E | F |

価格は各モジュールの定数の注記に出典と確認日（2026-09-25）がある。**`usd` 列はどれも見積り**で、
正は各社の請求の画面。

REST の 3 社は「1 本の接続に文を流し込む」口が無いので、文ごとにリクエストを投げる。LLM の生成を
止めないよう、`send()` は文を列に積むだけで戻り、別スレッドが**文の順に 1 本ずつ**投げて音を受ける
（順番を崩さないため、2 文目は 1 文目の音を受け終わってから投げる）。`t0` の前に課金の無い GET を 1 回投げて、
TCP と TLS を張っておく。

### 結果（5 問を 1 回ずつ。中央値）

同じ 5 問・同じ検索・同じ prompt・同じ Gemini（`gemini-3.6-flash`）。ElevenLabs は `eleven_turbo_v2_5`（2026-09-11）、
他の 3 社は 2026-09-25。質問ごとの表と読み方は `COMPARISON.md`。

| | ElevenLabs | Deepgram | OpenAI | Google |
|---|---:|---:|---:|---:|
| 最初の音まで | 3,796 ms | **3,462 ms** | 4,640 ms | 3,897 ms |
| TTS 区間 | 443 ms | **211 ms** | 1,200 ms | 1,386 ms |
| 言い終わりまで（並べて読まない） | 3,969 ms | 10,570 ms | 7,808 ms | 5,296 ms |
| 1 往復のドル（平均） | $0.0045 | $0.0029 | $0.0041 | **$0.0027** |

- **TTS 区間は Deepgram が一番短く、ElevenLabs が次**。OpenAI は最初のバイトが遅く、Google は 1 文ぶんの合成が
  終わるまで何も届かない（届け方の違いが数字に乗っている）
- **体感の遅延の大半は、どの社でも Gemini の最初のトークン**（2.1〜7.0 秒）。TTS を替えても 1 秒単位では速くならない
- 揃っていない条件: 声が社ごとに違う、ElevenLabs だけ 2 週間前に測った、読みの品質はまだ耳で比べていない

## プロバイダを足す手順

`voicelab/tts_path.py` の説明に同じ手順がある。見本は `voicelab/deepgram_path.py`（REST の社は 1 社 360〜460 行）。

1. `voicelab/<社>_path.py` を作り、TTS のクラスを書く。`custom_path.TtsStream` を満たし（`send` で文を積む、
   `finish` / `wait` / `close`、`sent_sentences` / `sent_chars` / `first_send_at`）、`with` で開けるようにする
   （`__enter__` で接続を温める）。音は `custom_path.AudioSink` に入れ、失敗は `error` に残す
2. `tts_path.TtsSpec` を 1 つ作る。構成キー（`path`。**一度決めたら綴りを変えない**）、`results/` の会社の段、
   書き起こしの記号、鍵とモデルの `.env` キー名、既定のモデル、出力の形とサンプルレート、料金の換算
   （文字数課金なら `tts_path.usd_per_chars`）、残高切れの見分け方
3. `run_scenario` と `describe_dry_run` は `tts_path` の同名の関数に `SPEC` を渡すだけにする
4. `tts_provider.register(TtsProvider(...))` で登録する。`key`・表示名・表の見出し（`B: 自前構成（<社>）`）・
   `required_env`・残高を 1 行で返す関数（読めないならそう言う文でよい）
5. `voicelab/cli.py` の import と `PROVIDER_MODULES` にモジュールを足す（import されないと登録されず、`--tts` に出ない）
6. `.env.example` に鍵とモデルの行を、`voicelab/config.py` の `ENV_KEYS` にキー名を足す
7. `tests/test_<社>_path.py` を足す（既存の社のテストが見本。通信はダミーに差し替える）
8. `python run.py run custom --tts <社> --dry-run` で確かめてから、1 問だけ流す

`runs.csv` の列、`report` の表、`credits` の表示は登録簿から引くので触らなくてよい。

## 任せる範囲の比較（構成 A / B / C）

A と C は ElevenLabs Agents Platform の上に作った。B は各部品を自分でつなぐ。

**何をプラットフォームに渡し、何を手元に置くか**が構成の正体で、遅延も費用も運用の面倒さも
そこから出てくる。

| | A: Agents Platform | B: 自前構成 | C: Knowledge Base |
|---|---|---|---|
| ノートの本文 | **手元**（`corpus/`） | **手元**（`corpus/`） | **ElevenLabs に預ける**（アップロード） |
| 検索の実装 | **手元**（`search.py` を道具として呼ばれる） | **手元**（`search.py` を先に呼ぶ） | **ElevenLabs 側**（内部の RAG） |
| 検索を呼ぶ判断 | LLM（道具を選ぶ） | こちら（毎回必ず引く） | ElevenLabs 側（`usage_mode: auto`） |
| LLM の呼び出し | **ElevenLabs 側** | **自分**（Gemini API） | **ElevenLabs 側** |
| 音声（TTS） | ElevenLabs 側 | 自分でつなぐ（[4 社から選ぶ](#tts-4-社の比較構成-b)） | ElevenLabs 側 |
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
- 検索の質は**揃わない**。A・B は同じ `search.py`（文字バイグラム）、C は ElevenLabs の
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
書き起こし（`results/elevenlabs/knowledge-base/transcripts/<id>_kb_<UTC>.txt`）に丸ごと残している。
これが空なら「速かったのは何も引かなかったから」を疑う。

```
## RAG（ElevenLabs 側の検索）
- rag_usage: {"usage_count": 1, "embedding_model": "multilingual_e5_large_instruct"}
```

## なぜ web app ではないのか

共有できる形の音声対話は `obsidian-vault-web` の `/voice`（ブラウザ + WebRTC）で既に動いている。
ここで欲しいのは触れるデモではなく**数字**なので、ブラウザの音声処理層を挟まない Python にしている。

## 状態

**A・B・C とも 5 問流して並べた（2026-09-11、`COMPARISON.md`）。B の TTS は 4 社で 5 問ずつ流した
（ElevenLabs は 2026-09-11、Deepgram・OpenAI・Google は 2026-09-25。後の 3 社は正誤が未入力）。**

- [x] 題材データ（`corpus/`）と質問集（`scenarios/questions.json`）
- [x] クレジット残高の記録（`voicelab/credits.py`）
- [x] 計測結果の記録と表の出力（`voicelab/metrics.py`）
- [x] ノートの検索（`voicelab/search.py`）と Agent 用の client tool
- [x] Agent とツールの作成・更新（`voicelab/agent_setup.py`）
- [x] A: Agents Platform で 1 往復する（`voicelab/agents_path.py`）
- [x] B: 検索 + Gemini + ストリーミング TTS で 1 往復する（`voicelab/custom_path.py`）
- [x] ノートを Knowledge Base に預けて索引を張る（`voicelab/kb_setup.py`）
- [x] C: Knowledge Base で 1 往復する（`voicelab/kb_path.py`）
- [x] B の TTS を差し替える口（`voicelab/tts_provider.py`、`voicelab/tts_path.py`）と Deepgram・OpenAI・Google
- [ ] B（Deepgram / OpenAI / Google）の正誤と読みを耳で確かめる

## 使い方

### 最初の 1 回だけ（環境を作る）

```powershell
uv venv                       # .venv を作る（python -m venv .venv でも同じ）
uv sync --extra dev           # 依存 + テスト用を入れる
copy .env.example .env        # ELEVENLABS_API_KEY と GEMINI_API_KEY を書く（他社も流すならその鍵も）
```

Agent の id は手で書かなくてよい。`setup-agent` が `ELEVENLABS_AGENT_ID`（A）を、
`setup-kb` が `ELEVENLABS_KB_AGENT_ID`（C）を、それぞれ `.env` の該当行だけ書き換える。

`.env` のキーと、どこが使うか（書き方の例と声の一覧は `.env.example`）:

| キー | 使うところ | 中身 |
|---|---|---|
| `ELEVENLABS_API_KEY` | A・B（ElevenLabs）・C | ElevenLabs の鍵。残高（`credits`）もこれで読む |
| `ELEVENLABS_AGENT_ID` / `ELEVENLABS_KB_AGENT_ID` | A / C | `setup-agent` / `setup-kb` が書く |
| `ELEVENLABS_VOICE_ID` / `ELEVENLABS_MODEL_ID` | A・B（ElevenLabs）・C | 声と TTS モデル |
| `VOICELAB_TTS_MODEL` | B（ElevenLabs） | B だけ TTS モデルを替えるとき（`eleven_v3` は stream-input で使えないため。`.env.example` には無い） |
| `VOICELAB_LLM` | 全部 | 応答生成の LLM（既定 `gemini-3.6-flash`） |
| `GEMINI_API_KEY` | B（どの TTS でも） | Gemini を自分で呼ぶ構成だけが使う |
| `DEEPGRAM_API_KEY` / `DEEPGRAM_TTS_MODEL` | B（Deepgram） | 鍵とモデル（＝声）。既定 `aura-2-izanami-ja`。`billing:read` があれば残高も読める |
| `OPENAI_API_KEY` / `OPENAI_TTS_MODEL` / `OPENAI_TTS_VOICE` | B（OpenAI） | 鍵・モデル・声。空なら `gpt-4o-mini-tts` / `marin` |
| `GOOGLE_TTS_API_KEY` / `GOOGLE_TTS_VOICE` | B（Google） | Cloud Text-to-Speech の API キーと声。空なら `ja-JP-Chirp3-HD-Kore` |

### 2 回目以降（uv を使わない）

`.venv` は**ごく普通の venv** で、uv 固有のものは入っていない。一度有効化すれば、
あとは素の Python のコマンドだけで完結する。

```powershell
.venv\Scripts\activate       # 有効化（プロンプトの頭に (elevenlabs) が付く）

python run.py credits                               # 残高（ElevenLabs と、鍵のある各社の 1 行）
python run.py scenarios                             # 質問集
python run.py run agents --scenario rollback        # A で 1 問（課金あり）
python run.py run custom --scenario rollback        # B（ElevenLabs）で 1 問（課金あり）
python run.py run custom --tts google --scenario rollback   # B（Google）で 1 問（Google に課金）
python run.py run kb --scenario rollback            # C で 1 問（課金あり）
python run.py run custom --tts openai --dry-run     # 接続せず確認だけ（課金なし）
python run.py setup-kb                              # C の登録・索引・Agent（課金なし）
python run.py report                                # 表を作る
pytest -q                                           # テスト
```

`--tts` に使えるのは `elevenlabs`（既定）/ `deepgram` / `openai` / `google`。
`run deepgram` のように社名を直接書く古い呼び方も通る（`run custom --tts deepgram` と同じ）。
`deactivate` で抜ける。

`--dry-run` は検索と設定だけを見せる。鍵は「あり / 未設定」だけ出し、値は出さない。例（Google）:

```
[dry-run] rollback: ロールバックはどうやるの
  LLM: gemini-3.6-flash（鍵 GEMINI_API_KEY: あり）
  TTS: Google ja-JP-Chirp3-HD-Kore / linear16_16000（鍵 GOOGLE_TTS_API_KEY: あり）
  期待するノート: デプロイ手順
  検索の結果（B と同じ。LLM に選ばせず、先にこれを渡す）:
    1. デプロイ手順 / ロールバック
    2. デプロイ手順 / デプロイ前の確認
    3. デプロイ手順 / 手順
  system prompt: 924 文字（B と同じもの）
  実行すると: 検索 → LLM をストリーミング → 文が確定するたびに /v1/text:synthesize へ 1 本ずつ投げ、1 文ぶんの音をまとめて受ける（REST はストリーミングしない）
```

### search.py は何をしているのか（ElevenLabs のコードが 1 行も無い理由）

`voicelab/search.py` は `corpus/` の 3 本を文字バイグラムで引くだけの関数で、
ElevenLabs にも Gemini にも依存していない。それが**そのまま両方の統合点**になっている。

- **A**: ElevenLabs の Agent に `search_notes` という **client tool** を登録してある。
  Agent が「ノートを引く」と判断すると、その呼び出しが会話の WebSocket でこちらに届き、
  `search.search_notes()` が手元で走って戻り値が LLM に渡る（`agents_path.py` の handler）。
  ツールがサーバ側実行（webhook）ではないので、公開 URL もトンネルも要らない
- **B**: LLM に選ばせず、質問が来たら**先に**同じ関数を呼んで、結果を prompt に添える。TTS がどの社でも同じ
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
| `python -m voicelab.kb_path rollback` | `kb_path.run_scenario()` | **あり** |
| `python -m voicelab.custom_path rollback` | `custom_path.run_scenario()`（B・ElevenLabs） | **あり** |
| `python -m voicelab.deepgram_path rollback` | `deepgram_path.run_scenario()`（`openai_path` / `google_path` も同じ形） | **あり**（その社） |

課金のある行は 1 往復ぶん課金される。**残高の見張りは `cli.py` にあるので、この叩き方では効かない。**

REPL からでも同じ。CLI を通さずに関数を直接呼べるよう、表示と引数解析は `cli.py` に、
処理は各モジュールに分けてある（テストも CLI を通さず関数を直接呼んでいる）。

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

- `uv venv` … `.venv` を作る。`python -m venv .venv` と同じ
- `uv sync` … `pyproject.toml` と `uv.lock` のとおりに `.venv` を**揃える**。
  足りないものを入れ、**余計なものを消す**。`--extra dev` を付け忘れると pytest が消えるのはこのため
- `uv add <パッケージ>` … `pyproject.toml` に 1 行足し、`uv.lock` を更新し、`.venv` に入れる。
  有効化した状態の `pip install` でも `.venv` には入るが、`pyproject.toml` は更新されない
- `uv run <コマンド>` … `.venv` を選んでから実行する。有効化していれば要らない

つまり **uv が要るのは環境を作るときと依存を足すときだけ**で、それ以外は素の Python でよい。

## 結果の置き場

2026-09-25 から `results/<会社>/<モデル>/{audio,raw,transcripts}/` に分けて置く。表（`runs.csv`・`report.md`）だけは
`results/` 直下に 1 つ。録音と書き起こしの名前は `<質問>_<構成キー>_<UTC>.wav` / `.txt`、`raw/` は ElevenLabs の会話の生 JSON、
録音は git に入れない。

| 会社 / モデル | 中身 |
|---|---|
| `elevenlabs/agents-platform/` | A。A・C は TTS モデルを自分で選ばないので、モデルの段に構成の名前を入れる |
| `elevenlabs/knowledge-base/` | C |
| `elevenlabs/eleven_turbo_v2_5/`、`elevenlabs/eleven_flash_v2_5/` | B（ElevenLabs） |
| `elevenlabs/samples/audio/` | 声とモデルの聞き比べ（`voice-sample_*`、`date-sample_*`、`model-sample_*` など） |
| `deepgram/aura-2-izanami-ja/` | B（Deepgram） |
| `openai/gpt-4o-mini-tts_marin/` | B（OpenAI）。モデルと声が別の設定なので `<モデル>_<声>` |
| `google/ja-JP-Chirp3-HD-Kore/` | B（Google）。声の名前がモデルを兼ねる |

構成キー（ファイル名と `runs.csv` の `path` 列）は `agents` / `custom`（B・ElevenLabs）/ `kb` / `deepgram` / `openai` / `google`。
`runs.csv` には 2026-09-25 に `usd`（ドルの見積り）と `model`（上のモデルの段）の列を足した。
旧い置き場（`results/{audio,raw,transcripts}/`）からの並べ替えは `scripts/migrate_results_20260925.py` で済ませてある（1 回きり）。

## 計測の定義

| 名前 | 意味 |
|---|---|
| `first_audio_ms` | 質問を送ってから、**最初の音の断片**が届くまで。体感の遅延はこれ |
| `reply_done_ms` | 質問を送ってから、**最後の音の断片**が届くまで |
| `credits` | ElevenLabs のクレジット。A・C は会話の `metadata.cost`（取れなければ 0 にして `note` に「費用未取得」）、B（ElevenLabs）は文字数からの見積り、他社は 0 |
| `usd` | ドルの見積り。B の各社は定価から（ElevenLabs は Flash / Turbo の API 料金 $0.05 / 1,000 字で換算）。A・C は会話課金で文字数から出せないので 0（表では「—」） |
| TTS 区間 | B だけ。書き起こしの「最初の音まで − TTS へ最初の文」。LLM のぶれを除いて **TTS だけを比べる数字** |

- 質問は**テキストで送る**（`send_user_message`）。マイクを使うと、部屋の雑音と無音判定の
  ばらつきがそのまま数字に乗る。代わりに、**この数字に音声認識の時間は含まれない**。
  B も同じくテキスト投入から測る
- A・C の「言い終わり」は、最初の音が来てから **1.5 秒**音が途切れたら、と決めている。サーバの終了イベント
  ではなく音の途切れで決めるのは、B でも同じ判定が書けるから。判定が違うと数字を並べられない
- 「言い終わり」は**最後の音が届いた時刻**で、再生し終わる時刻ではない。音の届き方（一括か、少しずつか）が
  社ごとに違うので、**B の TTS の社どうしでは並べて読まない**（`COMPARISON.md`）
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

> LLM は当初 `gemini-2.5-flash` だったが、2026-09-11 に Gemini API が「新規ユーザーには提供終了」（404）を返したため、A・B とも `gemini-3.6-flash` に揃えた。

`voicelab/custom_path.py`。直列のパイプラインで、A と同じ `Run` を返す。

```
質問テキスト → 検索（上位 3 件） → Gemini をストリーミング → 文が確定するたびに TTS へ → 音
```

### A との違い

| | A: Agents Platform | B: 自前構成 |
|---|---|---|
| 検索の呼び方 | **LLM がツールを選ぶ**（client tool） | **選ばせない。先に検索して結果を渡す** |
| ノートに無い質問 | LLM がツールを呼ばずに「見当たりません」 | それでも検索する。0 件を渡して LLM に言わせる |
| LLM | `gemini-3.6-flash`（ElevenLabs 側が動かす） | 同じ `gemini-3.6-flash`（**自分で呼ぶ**） |
| 費用の出方 | 会話の `metadata.cost` に全部込み（分単位） | TTS の**文字数**、Gemini は**トークン**。別勘定 |
| 面倒を見るもの | ほぼ無し | 文の切り出し・TTS の接続・受信スレッド・時刻の記録 |

ツール往復（LLM がツールを選ぶ → 検索 → LLM が答えを作る）が 1 回消えるぶん、B の方が
速いはず、という仮説を確かめるための構成。A の実測ではその往復が 1.5〜2 秒だった。

### 文単位のストリーミングである

**LLM の生成完了は待たない。** トークンを溜めて「。」「！」「？」で切り、文が 1 つ確定した
時点で TTS へ送る。半角の `.` では切らない（`v2.5` のような版番号で途中まで送ってしまうため）。
ここまではどの社の TTS でも同じ関数（`custom_path.run_pipeline` と `run_turn`）を通る。

ElevenLabs の TTS は `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input` に
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
質問テキストを渡した時刻で、**TTS の接続と Gemini の client 作成は `t0` より前**に済ませる
（A も接続後に質問を送っているため）。CSV に入らない内訳は書き起こしに書く。

- 検索 ms / LLM 最初のトークン ms / LLM 完了 ms / TTS へ最初の文 ms
- TTS に送った文の数と文字数
- Gemini の `usage_metadata`（入力・出力・思考・合計トークン）
- ElevenLabs 以外は、文ごとの応答時間（リクエストから最初のバイトまで。Google は応答が全部届くまで）

### 費用

B（ElevenLabs）の `credits` 列には **TTS に送った文字数 × 0.5** を入れる（flash / turbo 系を API から使ったときの
1 文字あたりのクレジット。根拠と出典は `voicelab/custom_path.py` の `CREDITS_PER_CHARACTER`）。
これは**見積り**で、正は実行前後の残高の差。ただし残高の反映は遅れる。

**Gemini の費用は TTS の費用に混ぜない。** トークン数として書き起こしに残す。
A ではこの分が会話の `cost` に溶けていて分けられない ―― そこも B との違い。

## A の最初の結果（2026-09-11、gemini-2.5-flash の頃）

全行は `results/runs.csv`。音声は `results/elevenlabs/agents-platform/audio/`（git には入れない）。

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
- 返答は全問ノートを根拠にしていて、出典のノート名を口頭で添えている。`correct` 列は**書き起こしの本文**で判定した（期待ノート名を出典として言っているか。無い質問は「見当たらない」と言っているか）。音声の長さと文字数の比（6〜8 字/秒）も全行で妥当で、途中で切れた録音は無い。人が聞いて確かめる価値があるのは読み方だけで、A の「レシピプロカル」（LLM の書いたカタカナが誤り）と「二千二十六年」（漢数字読み）が候補

## A と B を並べる（どちらも gemini-3.6-flash、2026-09-11、TTS は flash_v2_5）

同じ 5 問を、同じ検索関数・同じ声・同じ TTS モデル・同じ LLM で流した。A・C を v3 に揃えて測り直した最新の表は `COMPARISON.md`。

| 質問 | A 最初の音 | B 最初の音 | B の内訳: LLM 初トークン | A クレジット | B クレジット（TTS 見積り） |
|---|---:|---:|---:|---:|---:|
| デプロイ前の確認 | 2,479 ms | 3,460 ms | 3,133 ms | 100 | 55 |
| ロールバック | 2,653 ms | 6,224 ms | 5,925 ms | 100 | 50 |
| RRF とは | 2,633 ms | 4,229 ms | 3,869 ms | 98 | 50 |
| 5/12 の定例 | 1,871 ms | 2,845 ms | 2,397 ms | 88 | 50 |
| 来週の天気（無い） | 2,049 ms | 1,674 ms | 1,481 ms | 85 | 16 |

- **A の方が速く、ばらつきも小さい**（1.9〜2.7 秒）。B は 1.7〜6.2 秒で、ほぼ全部が **Gemini の最初のトークンまでの時間**。検索は 1〜2 ms、TTS は最初の文を送ってから 200〜300 ms で音が来る。つまり B の遅さは自前の作りではなく、公開 API 経由の Gemini（`thinking_level='low'` でも思考に 300 トークン前後使う）の応答速度
- A は ElevenLabs の中で同じモデルを呼んでいるはずだが、ツール往復（LLM → ツール → LLM）込みで B の単発呼び出しより速い。中で何をしているか（思考の抑え方、プロビジョニング、地理）はこちらからは見えない。**「同じモデル名でも、誰がどう呼ぶかで 1〜3 秒変わる」**が、この比較で一番大きい発見
- 費用の出方が違う。A は会話ごとに **85〜100 クレジット**（LLM 込み・通話時間ベース）。B は **TTS の文字数ぶん 16〜55 クレジット**＋Gemini のトークン（入力 450 前後 / 出力 60 前後 / 思考 300 前後）。合算すると B の方が安いが、二社に請求が分かれる
- 2.5-flash のときの A（上の表）は 2.7〜3.3 秒だったので、**A は 3.6-flash で 0.5〜1 秒速くなった**。ノートに無い質問は 2.5 ではツールを呼ばず 1.3 秒、3.6 では律儀にツールを呼んで 2.0 秒
- `results/report.md` の A の中央値は 2.5-flash と 3.6-flash の両方の行を含む（`runs.csv` に LLM の列が無いため）。分けて見るときは日時で切る

## C の疎通確認（1 問だけ、2026-09-11、TTS は flash_v2_5）

5 問の前に 1 問だけ流し、つながることと RAG が引かれること（`rag_usage` が `usage_count: 1` /
`multilingual_e5_large_instruct`）を確かめた。デプロイ前の確認で **1,039 ms・53 クレジット**（通話 4 秒）と、
同じ質問の A（2,479 ms・100 クレジット）より速く安く見えた。ただし 5 問流すと C の中央値は 2.9 秒で A より遅かった
（`COMPARISON.md`）。**1 サンプルで結論を出さない。** 返答はノートを根拠にし、出典のノート名を添え、数字も算用数字だった。

## 声と数字の読み（2026-09-11）

聞き比べの録音は `results/elevenlabs/samples/audio/` にある。

- 当初の声 Sarah は ElevenLabs の検証済み言語に日本語が無い。無料でも使える premade のうち **George / Alice / Jessica は日本語が検証済み**（`GET /v1/voices` の `verified_languages`）。同じ文を 4 声で読ませて聞き比べ（`voice-sample_*.wav`）、**Jessica** に切り替えた（`.env` の `ELEVENLABS_VOICE_ID`）
- 年号の読みが不自然だったので、prompt に「算用数字で書く、日付は月日、年は省く」を足した。**B は従うが A は従わない**: A の返答は「五月十二日」「二十件から十件」と漢数字のまま（B は「5月12日」「20から10」）。TTS がどの書き方を一番自然に読むかは `date-sample_*.wav`（Jessica、5 通り）で聞き比べる
- **原因は Agent の TTS 設定 `text_normalisation_type`** だった。既定の `system_prompt` は LLM に「数字を語で書け」と指示するので、日本語では漢数字（五月十二日、二十件）になり、TTS の読みが崩れる。`elevenlabs`（生成後に ElevenLabs 側で整形）にすると LLM は算用数字のまま書き（5月12日、20から10）、読みが直った。遅延の増加は測定誤差の範囲（2.06 秒）。`agent_setup.py` で固定
- B の「たどたどしさ」は文ごとに TTS へ送る作りに由来する可能性がある。同じ文を flash_v2_5 と multilingual_v2 で一括合成した比較が `model-sample_*.wav`
- 2026-09-11 16:30 UTC 以降の `runs.csv` の行は Jessica。それ以前は Sarah。18:00 UTC 以降の A は正規化 `elevenlabs`

## ElevenLabs を日本語で使う人へ（つまずいた点と対処）

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

ここは **A と C（会話）の話**。B は会話ではなく TTS なので、どの社でも消費は喋った文字数ぶん
（1 往復 $0.001〜0.006。ElevenLabs なら 50 クレジット前後）で、放置しても増えない。
残高の見張り（`MIN_CREDITS`）は ElevenLabs を使う A・B（ElevenLabs）・C で共通の門をくぐらせている。
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
