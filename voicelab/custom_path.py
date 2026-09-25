"""B: 自前構成（検索 → Gemini → ストリーミング TTS）で 1 往復して、遅延と費用を記録する。

**責務はここだけ**。「質問を 1 つ投げて、返事を最後まで受け取り、時刻と費用を残す」。
質問の選び方（1 問か 5 問か）、残高の見張り、表の更新は :mod:`voicelab.cli` の仕事。
戻り値は A（:mod:`voicelab.agents_path`）と**同じ** :class:`voicelab.metrics.Run` で、
WAV と書き起こしも同じ場所（``results/audio/`` と ``results/transcripts/``）に残す。

A との違い（これが比べたいもの）:

- A は「LLM がツールを選ぶ」。B は **LLM に選ばせず、先に検索して結果を渡す**。
  ノートに無い質問でも検索は走り、0 件の結果を渡して LLM に「見当たらない」と言わせる。
  ツール往復（LLM → ツール → LLM）が 1 回消えるぶん、B の方が速いはず、という仮説。
- A の LLM は ElevenLabs 側で動く（費用は会話の ``cost`` に混ざる）。B は同じ
  ``gemini-3.6-flash`` を**自分で呼ぶ**ので、Gemini の費用と ElevenLabs の TTS 費用が別々に出る。
- TTS は**文が確定するたびに送る**。LLM の生成完了を待たない。

計測の定義は A と揃える（README の「計測の定義」）。``t0`` は質問テキストを渡した時刻、
``first_audio_ms`` は最初の音声チャンクが届くまで、``reply_done_ms`` は最後のチャンクまで。
内訳（検索・LLM・TTS の各段）は CSV には入らないので書き起こしに書く。
"""

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Iterator, Protocol

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from . import search
from .agents_path import audio_window_ms, elapsed_ms, save_audio_file, save_transcript
from .config import AGENT_PROMPT_PATH, COMPANY_ELEVENLABS, DEFAULT_LLM, load_env, require, result_dirs
from . import credits as credits_api
from . import tts_provider
from .metrics import PATH_CUSTOM, Run

#: 文の切れ目。ここで切って TTS に送る。
#:
#: 半角の ``.`` を入れていないのは、``2.5`` や ``v2.5`` のような数字・版番号で
#: 途中まで送ってしまうため。読み上げ向けの日本語では実害がない。
SENTENCE_ENDINGS = ('。', '！', '？')

#: TTS に送るとき、文の末尾に足す区切り。ElevenLabs は「テキストは空白で終えること」を求める。
TTS_SEPARATOR = ' '

#: 出力フォーマット。A の録音（16kHz / 16bit / mono の生 PCM）と同じにして、
#: WAV の作り方も揃える。形式が違うと「言い終わり」の見え方が変わってしまう。
OUTPUT_FORMAT = 'pcm_16000'

#: TTS の WebSocket。``wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input``
WS_BASE = 'wss://api.elevenlabs.io'

#: LLM に許す出力の長さ。2〜3 文の読み上げに 300 トークンあれば足りる。
#: 思考トークンもこの枠に数えられる。gemini-3.6-flash は ``thinking_level='low'`` でも
#: 300 前後を思考に使うので、300 だと本文が 1 文で切れた（2026-09-11 に実測）。
#: 返答の長さは system prompt（2〜3 文、120 文字以内）で抑える。
MAX_OUTPUT_TOKENS = 1024

#: 言い回しのぶれを抑える。比較のたびに答えが変わると数字を並べられない。
TEMPERATURE = 0.3

#: ``isFinal`` が来るまで待つ上限（秒）。来なければタイムアウトとして記録する。
TTS_DONE_TIMEOUT_SECONDS = 60.0

#: TTS 1 文字あたりのクレジット（flash / turbo 系を **API から** 使ったとき）。
#:
#: 公式ドキュメントは「UI からの生成は 1 文字 1 クレジット。API からの生成は割引」とだけ書き、
#: 割引後の係数をクレジットでは示していない。API の料金表では Flash / Turbo が 1,000 文字
#: $0.05、Multilingual v2 が 1,000 文字 $0.10 で、Multilingual v2（＝ 1 文字 1 クレジット）の
#: ちょうど半額。したがって **1 文字 0.5 クレジット**とする。
#:
#: 出典（2026-09-11 確認）:
#:
#: - https://elevenlabs.io/docs/help-center/technical/what-models-do-you-offer-and-what-is-the-difference-between-them
#:   （「For UI generations, 1 character costs 1 credit... API generations are discounted」）
#: - https://elevenlabs.io/pricing/api （Flash/Turbo $0.05 / 1K chars、v2 Multilingual $0.10 / 1K chars）
#:
#: これは**見積り**。正は実行前後の残高の差（``voicelab credits``）で、ずれたらここを直す。
#: 残高への反映は遅れるので、1 回の実行では合わないことがある（README の「最初の結果」を参照）。
CREDITS_PER_CHARACTER = 0.5

#: ElevenLabs の TTS を **ドルに換算するときの単価**（Flash / Turbo の API 定価、1,000 文字あたり）。
#: 他社（Deepgram / OpenAI / Google …）と同じ単位で並べるためのもので、契約プランの実勢
#: （Starter は $5 / 30,000 クレジット）とは違う。出典は上と同じ https://elevenlabs.io/pricing/api
USD_PER_1K_CHARS = 0.05


def estimate_usd(sent_chars: int) -> float:
    """TTS に送った文字数からドルを見積もる（API 定価）。"""
    return sent_chars * USD_PER_1K_CHARS / 1000

#: A の指示のうち、B では意味を成さない行の目印。B に道具は無く、検索はもう済んでいる。
TOOL_LINE_MARKER = 'search_notes'

#: 検索結果を system prompt に添えるときの前置き。
#: A の「必ず search_notes を呼ぶ」「結果に無いことは言わない」を、B の言い方に置き換えたもの。
SEARCH_PREAMBLE = (
    '検索はこちらで済ませました。呼べる道具はありません。'
    '次の検索結果にあることだけを使って答え、記憶や推測で補ってはいけません。'
    '結果に無いことは「ノートには見当たりません」と言い、話を作りません。'
)

#: 0 件だったときの文言。「無い」と言わせるための指示をここで固定する。
SEARCH_EMPTY = (
    '検索はこちらで済ませましたが、結果は 0 件でした。呼べる道具はありません。'
    '「ノートには見当たりません」と答え、話を作らないでください。'
)


class CustomPathError(RuntimeError):
    """1 往復を始められなかった、または続けられなかった。"""


# --------------------------------------------------------------------------- 文の切り出し


class SentenceBuffer:
    """LLM のトークンを溜めて、文が終わったところで切り出す。

    **存在理由**は「LLM の生成完了を待たずに TTS へ送る」こと。トークンは
    ``デプロイ`` ``手順`` ``によると`` のように文の途中で届くので、文末の記号が来るまで
    ここで溜める。切り出しは文字単位で見る（日本語には分かち書きが無く、
    1 トークンに複数の文が入ることもあるため）。
    """

    def __init__(self, endings: Iterable[str] = SENTENCE_ENDINGS):
        self._endings = tuple(endings)
        self._buffer = ''

    def feed(self, text: str) -> list[str]:
        """トークンを 1 つ食わせて、**確定した文だけ**を返す（無ければ空）。"""
        done: list[str] = []
        for char in text:
            self._buffer += char
            if char in self._endings:
                sentence = self._buffer.strip()
                if sentence:
                    done.append(sentence)
                self._buffer = ''
        return done

    def flush(self) -> list[str]:
        """溜まっている残りを吐く。**文末の記号が無いまま終わった分**をここで拾う。"""
        sentence = self._buffer.strip()
        self._buffer = ''
        return [sentence] if sentence else []


def split_sentences(text: str, endings: Iterable[str] = SENTENCE_ENDINGS) -> list[str]:
    """完成した文字列を文に切る。:class:`SentenceBuffer` と同じ規則。"""
    buffer = SentenceBuffer(endings)
    return buffer.feed(text) + buffer.flush()


# --------------------------------------------------------------------------- system prompt


def build_system_prompt(
    hits: list[dict], *, base: str | None = None, prompt_path=AGENT_PROMPT_PATH
) -> str:
    """A の指示（``prompts/agent_system.txt``）を流用して、検索結果を添える。

    文言を A と共通にしておくのは、**返答の長さと語り口が違うと遅延も費用も比べられない**ため。
    「2〜3 文・120 文字以内」「出典のノート名を口頭で」「記号は読み上げない」は
    そのまま残す。

    落とすのは ``search_notes`` に触れる行だけ。B に道具は無いので、そのまま渡すと
    LLM が呼べない道具を探しにいく。同じ趣旨（記憶で答えない・無いものは無いと言う）は
    :data:`SEARCH_PREAMBLE` と :data:`SEARCH_EMPTY` が引き継ぐ。
    """
    base = prompt_path.read_text(encoding='utf-8').strip() if base is None else base.strip()
    base = '\n'.join(
        line for line in base.splitlines() if TOOL_LINE_MARKER not in line
    ).strip()
    if hits:
        tail = f'{SEARCH_PREAMBLE}\n{json.dumps({"results": hits}, ensure_ascii=False)}'
    else:
        tail = SEARCH_EMPTY
    return f'{base}\n\n{tail}'


# --------------------------------------------------------------------------- 費用


def estimate_credits(sent_chars: int) -> int:
    """TTS に送った文字数からクレジットを見積もる。

    ElevenLabs の TTS は**文字数**の課金で、会話（A）のような分単位ではない。
    Gemini の費用はここに混ぜない（クレジットではないため）。トークン数は書き起こしに残す。
    """
    return round(sent_chars * CREDITS_PER_CHARACTER)


# --------------------------------------------------------------------------- 音声の受け皿


class AudioSink:
    """届いた音声チャンクの**時刻**とバイト列を溜める。

    A の :class:`voicelab.agents_path.MeasuringAudioInterface` と同じ役割。鳴らさない。
    受信スレッドから呼ばれるが、``list.append`` だけなので錠は要らない。
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.chunks: list[bytes] = []
        self.times: list[float] = []
        self._clock = clock

    def add(self, audio: bytes) -> None:
        self.times.append(self._clock())
        self.chunks.append(audio)

    def pcm(self) -> bytes:
        """届いた断片をつないだ生 PCM。"""
        return b''.join(self.chunks)


# --------------------------------------------------------------------------- TTS


class TtsStream(Protocol):
    """文を送って音を受け取るもの。テストではダミーを差し込む。"""

    sent_sentences: int
    sent_chars: int
    first_send_at: float | None

    def send(self, sentence: str) -> None: ...

    def finish(self) -> None: ...

    def wait(self, timeout: float) -> bool: ...

    def close(self) -> None: ...

    # run_pipeline は ``with tts:`` で開く（開くのは t0 より前）
    def __enter__(self) -> 'TtsStream': ...

    def __exit__(self, *_exc) -> None: ...


class WebSocketTts:
    """``stream-input`` の WebSocket に**直接**つないで、文単位で送る。

    SDK の ``client.text_to_speech.convert_realtime`` を使わなかった理由（2026-09-11 に
    ``elevenlabs`` 2.67.0 の ``realtime_tts.py`` を読んで判断）:

    - 中の ``text_chunker`` が区切りに使う文字が ``. , ? ! ; : - ( ) [ ] }`` と半角空白だけで、
      **日本語の「。」を知らない**。文を渡しても切れ目と認識されず、最後まで溜め込まれる
    - その ``text_chunker`` は「次の断片が来て初めて 1 つ前を送る」作りなので、
      **最初の文が 1 文ぶん遅れて出る**。``first_audio_ms`` を測るのにこれは致命的
    - 戻り値が generator なので、音声チャンクの到着時刻が「こちらが next() を呼んだ時刻」に
      なる。受信スレッドで即座に時刻を打ちたい計測とかみ合わない

    そこで ``websockets.sync`` で直接つなぐ。送信は呼び出し側のスレッド、受信は別スレッド。
    ``websockets`` の同期実装は「送信と受信が別スレッド」なら安全（``recv`` の同時呼び出しだけが不可）。

    API キーは初期メッセージではなく**接続ヘッダ**（``xi-api-key``）で渡す。SDK もそうしていて、
    ドキュメントの初期メッセージのキー名（``xi-api-key`` / ``xi_api_key``）の揺れを避けられる。

    各文は ``{"text": "文 ", "flush": true}`` で送る。``flush`` を付けないと
    ``chunk_length_schedule``（既定 50 文字）ぶん溜まるまで生成が始まらず、
    短い返答では最初の音が遅れる。
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str,
        sink: AudioSink,
        output_format: str = OUTPUT_FORMAT,
        connect_fn: Callable[..., object] = ws_connect,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._sink = sink
        self._output_format = output_format
        self._connect_fn = connect_fn
        self._clock = clock
        self._socket = None
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self.sent_sentences = 0
        self.sent_chars = 0
        self.first_send_at: float | None = None
        self.error: str | None = None

    @property
    def url(self) -> str:
        return (
            f'{WS_BASE}/v1/text-to-speech/{self._voice_id}/stream-input'
            f'?model_id={self._model_id}&output_format={self._output_format}'
        )

    def open(self) -> 'WebSocketTts':
        """つないで初期メッセージを送り、受信スレッドを立てる。

        **t0 より前に呼ぶこと。** A も接続が済んでから質問を送っている。接続の時間を
        遅延に含めると、比べているものが変わってしまう。
        """
        try:
            self._socket = self._connect_fn(
                self.url, additional_headers={'xi-api-key': self._api_key}
            )
            self._socket.send(
                json.dumps(
                    {
                        'text': ' ',
                        'voice_settings': {'stability': 0.5, 'similarity_boost': 0.75},
                    }
                )
            )
        except Exception as exc:  # 接続の失敗はここで潰さない。呼び出し側が記録して止まる
            raise CustomPathError(f'TTS の WebSocket につなげませんでした: {exc}') from exc
        self._thread = threading.Thread(target=self._receive, daemon=True)
        self._thread.start()
        return self

    def _receive(self) -> None:
        """``isFinal`` が来るまで受け続ける。届いた瞬間に :class:`AudioSink` が時刻を打つ。"""
        try:
            while True:
                message = json.loads(self._socket.recv())
                audio = message.get('audio')
                if audio:
                    self._sink.add(base64.b64decode(audio))
                if message.get('isFinal'):
                    break
        except ConnectionClosed:
            pass  # 言い終わってサーバから閉じられる。異常ではない
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self._done.set()

    def send(self, sentence: str) -> None:
        """文が 1 つ確定したら即座に送る。**LLM の完了は待たない。**"""
        payload = sentence + TTS_SEPARATOR
        if self.first_send_at is None:
            self.first_send_at = self._clock()
        self.sent_sentences += 1
        self.sent_chars += len(payload)
        self._socket.send(json.dumps({'text': payload, 'flush': True}))

    def finish(self) -> None:
        """「もう送らない」を伝える（空文字）。残りの音声はこの後に届く。"""
        self._socket.send(json.dumps({'text': ''}))

    def wait(self, timeout: float) -> bool:
        """``isFinal``（か切断）まで待つ。時間切れなら False。"""
        return self._done.wait(timeout)

    def close(self) -> None:
        try:
            if self._socket is not None:
                self._socket.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> 'WebSocketTts':
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


# --------------------------------------------------------------------------- LLM


def chunk_text(chunk: object) -> str:
    """ストリームの断片から本文を取る。``None`` の回（usage だけの回）は空文字。"""
    return getattr(chunk, 'text', None) or ''


def chunk_usage(chunk: object) -> dict[str, int]:
    """断片に載っている ``usage_metadata`` を辞書にする（無ければ空）。

    Gemini は最後の断片に累計を載せてくる。費用は ElevenLabs のクレジットではないので
    ``Run.credits`` には混ぜず、書き起こしにトークン数として残す。
    """
    usage = getattr(chunk, 'usage_metadata', None)
    if usage is None:
        return {}
    keys = (
        'prompt_token_count',
        'candidates_token_count',
        'thoughts_token_count',
        'total_token_count',
    )
    found = {}
    for key in keys:
        value = getattr(usage, key, None)
        if value:
            found[key] = int(value)
    return found


def stream_gemini(
    question: str, system_prompt: str, *, api_key: str, model: str
) -> Iterator[object]:
    """``gemini-3.6-flash`` をストリーミングで呼ぶ。

    ``google.genai`` の import をここに閉じ込めているのは、``voicelab credits`` のような
    LLM と無関係なコマンドの起動を重くしないため。

    ``thinking_level='low'``（最小の思考）にしている理由: 2〜3 文の読み上げに推論は要らず、
    ``max_output_tokens`` が 300 しかないので、思考でこの枠を使い切って**本文が空のまま
    終わる**事故を避けたい。A 側の Agent も応答の速さを優先した設定なので、条件も揃う。
    ``thinking_budget=0`` は gemini-3.6-flash では 400（INVALID_ARGUMENT）になり、思考設定なしでは
    本文が空で返った（2026-09-11 に実測）。

    generator にしているのは、``genai.Client`` を**ストリームを読み終わるまで生かす**ため。
    クライアントをローカル変数で作って iterator だけ返すと、関数を抜けた時点でクライアントが
    回収されて HTTP 接続が閉じ、最初のチャンクを読む前に
    ``Cannot send a request, as the client has been closed`` で落ちる（2026-09-11 に実測）。
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    try:
        yield from client.models.generate_content_stream(
            model=model,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_level='low'),
            ),
        )
    finally:
        close = getattr(client, 'close', None)
        if callable(close):
            close()


# --------------------------------------------------------------------------- 1 往復


@dataclass
class Turn:
    """1 往復で測ったもの。CSV に入るのは一部で、残りは書き起こしに書く。

    :ivar search_ms: 検索が終わるまで（t0 から）。
    :ivar first_token_ms: LLM の最初のトークンが届くまで。来なければ None。
    :ivar llm_done_ms: LLM のストリームが終わるまで。
    :ivar tts_first_send_ms: 最初の文を TTS に送った時刻。
    :ivar first_audio_ms: 最初の音声チャンクが届くまで（**体感の遅延**。A と同じ定義）。
    :ivar reply_done_ms: 最後の音声チャンクが届くまで（A と同じ定義）。
    :ivar sent_chars: TTS に送った文字数。費用の元。
    :ivar usage: Gemini の ``usage_metadata``。
    :ivar timed_out: 音が来ない、または ``isFinal`` を待ちきれなかった。
    """

    question: str
    hits: list[dict]
    sentences: list[str] = field(default_factory=list)
    reply: str = ''
    search_ms: int = 0
    first_token_ms: int | None = None
    llm_done_ms: int | None = None
    tts_first_send_ms: int | None = None
    first_audio_ms: int = 0
    reply_done_ms: int = 0
    sent_chars: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    timed_out: bool = False
    error: str | None = None
    pcm: bytes = field(default=b'', repr=False)

    @property
    def credits(self) -> int:
        return estimate_credits(self.sent_chars)

    @property
    def top_note(self) -> str | None:
        return self.hits[0]['note'] if self.hits else None


def run_turn(
    question: str,
    *,
    hits: list[dict],
    llm_chunks: Iterable[object],
    tts: TtsStream,
    sink: AudioSink,
    t0: float,
    search_done_at: float,
    clock: Callable[[], float] = time.perf_counter,
    wait_seconds: float = TTS_DONE_TIMEOUT_SECONDS,
) -> Turn:
    """検索済みの結果と LLM のストリームを受け取り、文を TTS に流して :class:`Turn` を返す。

    **ここには外部との接続が無い**（LLM も TTS も引数で受け取る）。1 往復の流れ
    ――文が確定したら即座に送る、最後に空文字で締める、時刻をどこで打つか――を
    ネットワークもクレジットも使わずに確かめられるようにするため。
    """
    turn = Turn(question=question, hits=list(hits))
    turn.search_ms = elapsed_ms(t0, search_done_at)
    buffer = SentenceBuffer()
    parts: list[str] = []
    first_token_at: float | None = None

    for chunk in llm_chunks:
        turn.usage.update(chunk_usage(chunk))
        text = chunk_text(chunk)
        if not text:
            continue
        if first_token_at is None:
            first_token_at = clock()
        parts.append(text)
        for sentence in buffer.feed(text):
            tts.send(sentence)
            turn.sentences.append(sentence)

    llm_done_at = clock()
    for sentence in buffer.flush():
        tts.send(sentence)
        turn.sentences.append(sentence)
    tts.finish()
    finished = tts.wait(wait_seconds)

    turn.reply = ''.join(parts).strip()
    turn.first_token_ms = None if first_token_at is None else elapsed_ms(t0, first_token_at)
    turn.llm_done_ms = elapsed_ms(t0, llm_done_at)
    turn.tts_first_send_ms = (
        None if tts.first_send_at is None else elapsed_ms(t0, tts.first_send_at)
    )
    turn.sent_chars = tts.sent_chars
    turn.first_audio_ms, turn.reply_done_ms = audio_window_ms(t0, sink.times)
    turn.pcm = sink.pcm()
    turn.timed_out = not finished or not sink.times
    turn.error = getattr(tts, 'error', None)
    return turn


# --------------------------------------------------------------------------- 記録


def build_note(scenario: dict, turn: Turn) -> str:
    """CSV の 1 列に収まる長さで、あとから見て困らないだけのことを書く（A と同じ調子）。"""
    parts = [
        f'検索1位={turn.top_note}',
        f'期待={scenario.get("expected_note")}',
        f'LLM初トークン{turn.first_token_ms if turn.first_token_ms is not None else "-"}ms',
        f'文{len(turn.sentences)}本{turn.sent_chars}字',
        f'返答={turn.reply.replace(chr(10), " ")[:40] or "（なし）"}',
    ]
    if turn.timed_out:
        parts.append('音声が来ずタイムアウト')
    if turn.error:
        parts.append(f'TTS エラー={turn.error}')
    return ' / '.join(parts)


def render_transcript(scenario: dict, turn: Turn, *, voice_id: str, model_id: str, llm: str) -> str:
    """1 往復を、音を聞かなくても追える形の文章にする。

    CSV には入らない**内訳**（検索・LLM・TTS の各段と Gemini のトークン）はここにしか無い。
    A と B のどちらが速いかではなく、**どこで時間を使っているか**を見るための行。

    節の並びは :func:`render_turn_transcript` が持つ。ここで決めるのは B に固有の 2 か所
    （構成の行と、ElevenLabs のクレジットの見積り）だけ。
    """
    return render_turn_transcript(
        scenario,
        turn,
        config_line=f'B（検索 → {llm} → ElevenLabs {model_id} / 声 {voice_id}）',
        tts_cost_line=(
            f'- ElevenLabs クレジット（見積り）: {turn.credits}'
            f'（{turn.sent_chars} 字 × {CREDITS_PER_CHARACTER}）'
        ),
        audio_format=OUTPUT_FORMAT,
    )


def render_turn_transcript(
    scenario: dict, turn: Turn, *, config_line: str, tts_cost_line: str, audio_format: str
) -> str:
    """B と D が共有する書き起こしの本体。

    D（:mod:`voicelab.deepgram_path`）は TTS だけを差し替えた構成なので、節の並び
    （検索・返答・送った文・時刻・費用・その他）は B と**同じ**にしておく。
    違うのは「構成」の行と「TTS の費用」の行だけで、呼び出し側が渡す。
    """
    usage = turn.usage
    lines = [
        f'質問: {turn.question}（{scenario["id"]}）',
        f'期待するノート: {scenario.get("expected_note")}',
        f'構成: {config_line}',
        '',
        '## 検索（LLM に選ばせず先に実行）',
    ]
    if turn.hits:
        for index, hit in enumerate(turn.hits, start=1):
            lines.append(f'- {index}. {hit["note"]} / {hit["heading"] or "（見出しなし）"}')
    else:
        lines.append('- 0 件（LLM に「見当たりません」と言わせる）')

    lines += [
        '',
        '## 返答',
        turn.reply or '（なし）',
        '',
        '## TTS に送った文',
        *(f'- {s}' for s in turn.sentences or ['（なし）']),
        '',
        '## 時刻（t0 = 質問テキストを渡した時刻）',
        f'- 検索: {turn.search_ms} ms',
        f'- LLM 最初のトークン: {turn.first_token_ms if turn.first_token_ms is not None else "（来ず）"} ms',
        f'- LLM 完了: {turn.llm_done_ms} ms',
        f'- TTS へ最初の文: {turn.tts_first_send_ms if turn.tts_first_send_ms is not None else "（送らず）"} ms',
        f'- 最初の音まで: {turn.first_audio_ms} ms',
        f'- 言い終わりまで: {turn.reply_done_ms} ms',
        '',
        '## 費用',
        f'- TTS に送った文: {len(turn.sentences)} 本 / {turn.sent_chars} 文字',
        tts_cost_line,
        f'- Gemini トークン: 入力 {usage.get("prompt_token_count", "?")}'
        f' / 出力 {usage.get("candidates_token_count", "?")}'
        f' / 思考 {usage.get("thoughts_token_count", 0)}'
        f' / 合計 {usage.get("total_token_count", "?")}'
        '（ElevenLabs のクレジットではない。credits 列には入れない）',
        '',
        '## その他',
        f'- 音声: {len(turn.pcm):,} バイト（{audio_format}）',
        f'- タイムアウト: {"あり" if turn.timed_out else "なし"}',
        f'- TTS エラー: {turn.error or "なし"}',
    ]
    return '\n'.join(lines) + '\n'


def describe_dry_run(scenario: dict, env: dict[str, str] | None = None) -> str:
    """接続せずに、何をするつもりかと検索結果、送る system prompt の先頭を見せる。

    クレジットも Gemini のトークンも 1 も使わずに「検索が期待のノートを引けるか」
    「LLM に渡る材料が想定どおりか」を確かめられるようにするため。
    """
    env = load_env() if env is None else env
    hits = search.search(scenario['text'])
    prompt = build_system_prompt(hits)
    head = prompt[:240].replace('\n', '\n      ')
    lines = [
        f'[dry-run] {scenario["id"]}: {scenario["text"]}',
        f'  LLM: {env.get("VOICELAB_LLM") or DEFAULT_LLM}'
        f'（鍵 GEMINI_API_KEY: {"あり" if env.get("GEMINI_API_KEY") else "未設定"}）',
        f'  声/TTS モデル: {env.get("ELEVENLABS_VOICE_ID", "")} / {env.get("ELEVENLABS_MODEL_ID", "")}'
        f' / {OUTPUT_FORMAT}',
        f'  期待するノート: {scenario.get("expected_note")}',
        '  検索の結果（LLM に選ばせず、先にこれを渡す）:',
    ]
    if hits:
        for index, hit in enumerate(hits, start=1):
            lines.append(f'    {index}. {hit["note"]} / {hit["heading"] or "（見出しなし）"}')
    else:
        lines.append('    （0 件。LLM に「ノートには見当たりません」と言わせる）')
    lines += [
        f'  system prompt（{len(prompt)} 文字。先頭 240 字）:',
        f'      {head}...',
        '  実行すると: 検索 → LLM をストリーミング → 文が確定するたびに TTS へ送り、'
        'isFinal まで音を受ける',
    ]
    return '\n'.join(lines)


def run_pipeline(
    question: str, *, tts, sink: AudioSink, gemini_key: str, llm: str
) -> Turn:
    """TTS を開いてから ``t0`` を打ち、検索 → Gemini → TTS を 1 往復流す。

    **B と D の共通部分**。D（:mod:`voicelab.deepgram_path`）は TTS だけを差し替えた構成で、
    ``t0`` の打ち方・検索・Gemini の呼び方がずれると数字を並べられなくなるので、
    ここを 1 か所にしてある。``tts`` は :class:`TtsStream` を満たし、``with`` で
    開いて閉じられるもの（開くのは ``t0`` より**前**）。
    """
    with tts:
        t0 = time.perf_counter()
        hits = search.search(question)
        search_done_at = time.perf_counter()
        chunks = stream_gemini(
            question,
            build_system_prompt(hits),
            api_key=gemini_key,
            model=llm,
        )
        return run_turn(
            question,
            hits=hits,
            llm_chunks=chunks,
            tts=tts,
            sink=sink,
            t0=t0,
            search_done_at=search_done_at,
        )


def run_scenario(scenario: dict, *, save_audio: bool = True, env: dict[str, str] | None = None) -> Run:
    """質問を 1 つ投げて 1 往復し、:class:`voicelab.metrics.Run` を返す。

    **クレジットと Gemini のトークンを消費する。** 呼ぶ前に残高を確かめること
    （``voicelab run custom`` がやる）。

    接続（TTS の WebSocket と Gemini の client）は ``t0`` より**前**に済ませる。
    A も接続後に質問を送っているので、そこを含めると比べているものが変わる。

    :raises CustomPathError: TTS につなげなかった。
    :raises voicelab.config.ConfigError: 鍵が足りない。
    """
    env = load_env() if env is None else env
    api_key = require('ELEVENLABS_API_KEY', env)
    gemini_key = require('GEMINI_API_KEY', env)
    voice_id = require('ELEVENLABS_VOICE_ID', env)
    model_id = (env.get('VOICELAB_TTS_MODEL') or require('ELEVENLABS_MODEL_ID', env))
    llm = env.get('VOICELAB_LLM') or DEFAULT_LLM

    sink = AudioSink()
    tts = WebSocketTts(
        api_key=api_key, voice_id=voice_id, model_id=model_id, sink=sink
    )
    turn = run_pipeline(scenario['text'], tts=tts, sink=sink, gemini_key=gemini_key, llm=llm)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dirs = result_dirs(COMPANY_ELEVENLABS, model_id)
    if save_audio and turn.pcm:
        save_audio_file(turn.pcm, scenario['id'], stamp, dirs.audio, label=PATH_CUSTOM)
    save_transcript(
        render_transcript(scenario, turn, voice_id=voice_id, model_id=model_id, llm=llm),
        scenario['id'],
        stamp,
        dirs.transcripts,
        label=PATH_CUSTOM,
    )

    return Run(
        scenario_id=scenario['id'],
        path=PATH_CUSTOM,
        first_audio_ms=turn.first_audio_ms,
        reply_done_ms=turn.reply_done_ms,
        credits=turn.credits,
        usd=round(estimate_usd(turn.sent_chars), 6),
        model=dirs.model,
        correct=None,  # 人が音を聞いて判定する
        note=build_note(scenario, turn),
    )


def describe_balance(env: dict[str, str]) -> str:
    """ElevenLabs の残クレジットを 1 行で。読めなければその旨（止まらない）。"""
    try:
        return credits_api.read_subscription(env['ELEVENLABS_API_KEY']).describe()
    except KeyError:
        return 'ElevenLabs: 鍵 ELEVENLABS_API_KEY が未設定'
    except credits_api.CreditsError as exc:
        return f'ElevenLabs: 残高は確かめられませんでした（{exc}）'


#: B の既定の TTS（ElevenLabs）を登録簿に載せる。``run custom``（``--tts elevenlabs`` 省略時）はこれ。
#: ``key`` が ``custom`` なのは、CSV の ``path`` 列の綴りを 2026-09-11 の行から変えないため。
PROVIDER = tts_provider.register(
    tts_provider.TtsProvider(
        key=PATH_CUSTOM,
        name='ElevenLabs',
        label='B: 自前構成（ElevenLabs）',
        required_env=('ELEVENLABS_API_KEY', 'ELEVENLABS_VOICE_ID', 'GEMINI_API_KEY'),
        run_scenario=run_scenario,
        describe_dry_run=describe_dry_run,
        describe_balance=describe_balance,
        uses_elevenlabs_credits=True,
        cost_hint='ElevenLabs クレジット（見積り）',
        after_run_hints=(
            'Gemini のトークンは results/elevenlabs/<モデル>/transcripts/ に残している（クレジットとは別勘定）。',
        ),
    )
)


__all__ = [
    'AudioSink',
    'CREDITS_PER_CHARACTER',
    'PROVIDER',
    'USD_PER_1K_CHARS',
    'CustomPathError',
    'SentenceBuffer',
    'Turn',
    'WebSocketTts',
    'build_note',
    'build_system_prompt',
    'describe_balance',
    'describe_dry_run',
    'estimate_credits',
    'estimate_usd',
    'render_transcript',
    'render_turn_transcript',
    'run_pipeline',
    'run_scenario',
    'run_turn',
    'split_sentences',
    'stream_gemini',
]


if __name__ == '__main__':
    # python -m voicelab.custom_path rollback
    # **クレジットと Gemini のトークンを消費する。** 残高の見張りは cli 側にある。
    import sys

    from .config import find_scenario

    print(run_scenario(find_scenario(sys.argv[1] if len(sys.argv) > 1 else 'rollback')))
