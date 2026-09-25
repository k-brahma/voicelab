"""B の TTS を **Deepgram（Aura-2）** に差し替えた構成。

検索・Gemini の呼び方・``t0`` の打ち方・文の切り出し・時刻の記録は B の
:func:`~voicelab.custom_path.run_pipeline` と :func:`~voicelab.custom_path.run_turn` を
**そのまま使う**。1 往復の流し方と記録は :mod:`voicelab.tts_path` にあり、ここに置くのは
Deepgram に固有の 3 つ（:class:`DeepgramTts` の送受信、料金、残高）だけ。

B（ElevenLabs）との違い（これが比べたいもの）:

- TTS が ElevenLabs の ``stream-input``（WebSocket、1 本の接続に文を流し込む）から、
  Deepgram の ``POST /v1/speak``（REST、**1 文 1 リクエスト**）に変わる。応答本文は
  ストリーミングで読み、最初のバイトが届いた瞬間に時刻を打つ
- 出力は ``linear16`` / 16kHz / ``container=none``（生 PCM）。B の ``pcm_16000`` と同じ形で、
  WAV の作り方も B と同じ関数を使う
- 費用は ElevenLabs のクレジットではなく**ドル**（Aura-2 は 1,000 文字 $0.030）。
  ``credits`` 列は 0、ドルは ``usd`` 列と ``note`` 列と書き起こしに書く

日本語の声は 2026-09-25 に ``GET /v1/models`` で確かめて Aura-2 の 5 声だけ
（:data:`voicelab.config.DEFAULT_DEEPGRAM_TTS_MODEL` の注記）。Flux の TTS は一覧に無かった。
"""

import queue
import threading
import time
from typing import Callable
from urllib.parse import urlencode

import httpx

from . import credits, tts_path, tts_provider
from .config import DEFAULT_DEEPGRAM_TTS_MODEL
from .custom_path import AudioSink, CustomPathError, Turn
from .metrics import PATH_DEEPGRAM

#: Deepgram の REST の根。TTS は ``/v1/speak``、モデル一覧は ``/v1/models``。
API_BASE = 'https://api.deepgram.com'

#: ``results/deepgram/<モデル>/`` の会社の段。
COMPANY = 'deepgram'

#: 出力の形。B の ``pcm_16000``（16kHz / 16bit / mono の生 PCM）に揃える。
#: ``container=none`` を付けないと WAV のヘッダが先頭に付き、文ごとにつなぐと雑音になる。
ENCODING = 'linear16'
SAMPLE_RATE = 16000
CONTAINER = 'none'

#: 書き起こしと dry-run に出す出力形式の名前。
OUTPUT_FORMAT = f'{ENCODING}_{SAMPLE_RATE}'

#: Aura-2 の料金（Pay As You Go、1,000 文字あたりのドル）。
#:
#: 出典: https://deepgram.com/pricing （2026-09-25 確認。Aura-2 は PAYG $0.030 / Growth $0.027、
#: Aura-1 は PAYG $0.0150）。これは**見積り**で、正は残高の差（``voicelab credits``、鍵に
#: ``billing:read`` が要る）か Deepgram の使用量の画面。
USD_PER_1K_CHARS = 0.030

#: 1 リクエストの待ち上限（秒）。接続・読み取りとも。
REQUEST_TIMEOUT_SECONDS = 30.0


class DeepgramPathError(CustomPathError):
    """1 往復を始められなかった、または続けられなかった。

    :class:`voicelab.custom_path.CustomPathError` を継いでいるのは、CLI の受け口を
    増やさずに B と同じ扱いにするため。
    """


# --------------------------------------------------------------------------- 費用


def estimate_usd(sent_chars: int, usd_per_1k: float = USD_PER_1K_CHARS) -> float:
    """TTS に送った文字数からドルを見積もる。Deepgram の TTS は**文字数**の課金。"""
    return tts_path.usd_per_chars(sent_chars, usd_per_1k)


# --------------------------------------------------------------------------- TTS


def speak_url(model: str, *, base: str = API_BASE) -> str:
    """``POST /v1/speak`` の URL。モデル（＝声）と出力の形をクエリで渡す。"""
    params = {
        'model': model,
        'encoding': ENCODING,
        'sample_rate': SAMPLE_RATE,
        'container': CONTAINER,
    }
    return f'{base}/v1/speak?{urlencode(params)}'


def speak_body(sentence: str) -> dict:
    """``/v1/speak`` に送る本文。B と違って末尾の空白は要らない（ElevenLabs 固有の決まり）。"""
    return {'text': sentence}


def _make_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        headers={'Authorization': f'Token {api_key}'},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


class DeepgramTts:
    """文が確定するたびに ``POST /v1/speak`` を投げ、応答をストリーミングで受ける。

    :class:`voicelab.custom_path.TtsStream` を満たすので、B の ``run_turn`` にそのまま渡せる。

    **1 文 1 リクエスト**にしている。REST には B の ``stream-input`` のような「1 本の接続に
    文を流し込む」口が無いため。LLM の生成を止めないよう、``send()`` は文を列に積むだけで
    すぐ戻り、送信と受信は別スレッドが**文の順に 1 本ずつ**行う（音の順番を崩さないため）。
    そのぶん 2 文目以降は前の文の音を受け終わってから投げるので、``reply_done_ms`` は
    B より伸びうる。``first_audio_ms`` には影響しない。

    ``open()`` で 1 度 ``GET /v1/models``（課金なし）を投げて、TCP と TLS の接続を
    ``t0`` より**前**に張っておく。B が WebSocket の接続を ``t0`` 前に済ませているのと
    揃えるため。これをしないと、最初の文の遅延に接続の時間が乗る。
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        sink: AudioSink,
        client_factory: Callable[[str], httpx.Client] = _make_client,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._api_key = api_key
        self._model = model
        self._sink = sink
        self._client_factory = client_factory
        self._clock = clock
        self._client: httpx.Client | None = None
        self._queue: 'queue.Queue[str | None]' = queue.Queue()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self.sent_sentences = 0
        self.sent_chars = 0
        self.first_send_at: float | None = None
        #: 文ごとの「リクエストを投げてから最初のバイトまで」（ms）。書き起こしに書く。
        self.ttfb_ms: list[int] = []
        self.error: str | None = None

    @property
    def url(self) -> str:
        return speak_url(self._model)

    def open(self) -> 'DeepgramTts':
        """接続を温めて、送受信のスレッドを立てる。**t0 より前に呼ぶこと。**"""
        try:
            self._client = self._client_factory(self._api_key)
            response = self._client.get(f'{API_BASE}/v1/models')
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepgramPathError(
                f'Deepgram が HTTP {exc.response.status_code} を返しました（鍵を確かめる）'
            ) from exc
        except httpx.HTTPError as exc:
            raise DeepgramPathError(f'Deepgram に接続できませんでした: {type(exc).__name__}') from exc
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()
        return self

    def _work(self) -> None:
        """列から文を取り、1 本ずつ投げて音を受ける。届いた瞬間に :class:`AudioSink` が時刻を打つ。"""
        try:
            while True:
                sentence = self._queue.get()
                if sentence is None:
                    break
                started = self._clock()
                with self._client.stream('POST', self.url, json=speak_body(sentence)) as response:
                    if response.status_code >= 400:
                        body = response.read().decode('utf-8', errors='replace')
                        self.error = f'HTTP {response.status_code}: {body[:200]}'
                        break
                    first = True
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        self._sink.add(chunk)
                        if first:
                            self.ttfb_ms.append(round((self._sink.times[-1] - started) * 1000))
                            first = False
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self._done.set()

    def send(self, sentence: str) -> None:
        """文が 1 つ確定したら列に積む。**LLM の完了は待たない。**"""
        if self.first_send_at is None:
            self.first_send_at = self._clock()
        self.sent_sentences += 1
        self.sent_chars += len(sentence)
        self._queue.put(sentence)

    def finish(self) -> None:
        """「もう送らない」を伝える。列に残った文はこの後も順に処理される。"""
        self._queue.put(None)

    def wait(self, timeout: float) -> bool:
        """最後の文の音を受け終わる（か失敗する）まで待つ。時間切れなら False。"""
        return self._done.wait(timeout)

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)  # finish() を経ずに抜けたときの止め
            self._thread.join(timeout=2)
            self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> 'DeepgramTts':
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


def is_payment_required(error: str | None) -> bool:
    """TTS のエラーが残高切れ（HTTP 402）か。

    残高が無いと ``/v1/speak`` が ``402 ASR_PAYMENT_REQUIRED`` を返す（``/v1/models`` は通る）。
    2026-09-25 に、2024 年の無料枠が失効したアカウントで実際にこうなった。
    """
    return bool(error) and error.startswith('HTTP 402')


# --------------------------------------------------------------------------- 1 往復（共通ランナーに渡すもの）


def _make_tts(api_key: str, model: str, sink: AudioSink) -> DeepgramTts:
    # モジュールの属性を毎回引くのは、テストが ``DeepgramTts`` を差し替えられるようにするため
    return DeepgramTts(api_key=api_key, model=model, sink=sink)


def _pricing_line(turn: Turn) -> str:
    return (
        f'- Deepgram（見積り）: ${estimate_usd(turn.sent_chars):.5f}'
        f'（{turn.sent_chars} 字 × ${USD_PER_1K_CHARS} / 1,000 字）。ElevenLabs のクレジットは 0'
    )


def _extra_transcript(tts: object) -> list[str]:
    ttfb = getattr(tts, 'ttfb_ms', None) or []
    lines = ['## Deepgram の文ごとの応答（リクエストから最初のバイトまで）']
    lines += [f'- {index}. {ms} ms' for index, ms in enumerate(ttfb, start=1)] or ['- （なし）']
    return lines


SPEC = tts_path.TtsSpec(
    path=PATH_DEEPGRAM,
    company=COMPANY,
    name='Deepgram',
    letter='D',
    api_key_env='DEEPGRAM_API_KEY',
    model_env='DEEPGRAM_TTS_MODEL',
    default_model=DEFAULT_DEEPGRAM_TTS_MODEL,
    output_format=OUTPUT_FORMAT,
    sample_rate=SAMPLE_RATE,
    make_tts=_make_tts,
    estimate_usd=lambda turn: estimate_usd(turn.sent_chars),
    pricing_line=_pricing_line,
    is_payment_required=is_payment_required,
    payment_message='残高がありません（HTTP 402）。クレジットを足してから流し直す。results/ には何も書いていません',
    extra_transcript=_extra_transcript,
    dry_run_action='検索 → LLM をストリーミング → 文が確定するたびに /v1/speak へ 1 本ずつ投げ、音を受ける',
    error_type=DeepgramPathError,
)


def build_deepgram_note(scenario: dict, turn: Turn) -> str:
    """B の ``note`` にドルを足したもの。``credits`` 列が 0 なので、費用はここと ``usd`` 列で読む。"""
    return tts_path.build_usd_note(SPEC, scenario, turn)


class _TtfbHolder:
    """``render_transcript`` に文ごとの応答時間だけ渡すための入れ物。"""

    def __init__(self, ttfb_ms: list[int] | None):
        self.ttfb_ms = ttfb_ms or []


def render_transcript(
    scenario: dict, turn: Turn, *, model: str, llm: str, ttfb_ms: list[int] | None = None
) -> str:
    """B と同じ並びの書き起こしに、Deepgram の文ごとの応答時間を足す。"""
    return tts_path.render_transcript(
        SPEC, scenario, turn, model=model, llm=llm, tts=_TtfbHolder(ttfb_ms)
    )


def describe_dry_run(scenario: dict, env: dict[str, str] | None = None) -> str:
    """接続せずに、何をするつもりかと検索結果を見せる（課金なし）。"""
    return tts_path.describe_dry_run(SPEC, scenario, env)


def run_scenario(scenario: dict, *, save_audio: bool = True, env: dict[str, str] | None = None):
    """質問を 1 つ投げて 1 往復し、:class:`voicelab.metrics.Run` を返す。

    **Deepgram の残高と Gemini のトークンを消費する。**

    :raises DeepgramPathError: Deepgram につなげなかった、または残高切れ（HTTP 402。記録しない）。
    :raises voicelab.config.ConfigError: 鍵が足りない。
    """
    return tts_path.run_scenario(SPEC, scenario, save_audio=save_audio, env=env)


def describe_balance(env: dict[str, str]) -> str:
    """残高を 1 行で。**読めなくても止まらない**（403 は鍵の権限不足、空配列は残高なし）。"""
    try:
        return credits.read_deepgram_balance(env['DEEPGRAM_API_KEY']).describe()
    except KeyError:
        return 'Deepgram: 鍵 DEEPGRAM_API_KEY が未設定'
    except credits.CreditsError as exc:
        return f'Deepgram: 残高は確かめられませんでした（{exc}）'


PROVIDER = tts_provider.register(
    tts_provider.TtsProvider(
        key=PATH_DEEPGRAM,
        name='Deepgram',
        label='B: 自前構成（Deepgram）',
        required_env=('DEEPGRAM_API_KEY', 'GEMINI_API_KEY'),
        run_scenario=run_scenario,
        describe_dry_run=describe_dry_run,
        describe_balance=describe_balance,
        cost_hint='ElevenLabs クレジット 0（ドルは usd 列）',
        after_run_hints=(
            '文ごとの Deepgram の応答時間と Gemini のトークンは results/deepgram/<モデル>/transcripts/ にある。',
        ),
    )
)


__all__ = [
    'COMPANY',
    'DeepgramPathError',
    'DeepgramTts',
    'PROVIDER',
    'SPEC',
    'USD_PER_1K_CHARS',
    'build_deepgram_note',
    'describe_balance',
    'describe_dry_run',
    'estimate_usd',
    'is_payment_required',
    'render_transcript',
    'run_scenario',
    'speak_body',
    'speak_url',
]


if __name__ == '__main__':
    # python -m voicelab.deepgram_path rollback
    # **Deepgram の残高と Gemini のトークンを消費する。**
    import sys

    from .config import find_scenario

    print(run_scenario(find_scenario(sys.argv[1] if len(sys.argv) > 1 else 'rollback')))
