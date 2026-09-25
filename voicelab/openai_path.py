"""B の TTS を **OpenAI（gpt-4o-mini-tts）** に差し替えた構成。

検索・Gemini の呼び方・``t0`` の打ち方・文の切り出し・時刻の記録は B の
:func:`~voicelab.custom_path.run_pipeline` と :func:`~voicelab.custom_path.run_turn` を
**そのまま使う**。1 往復の流し方と記録は :mod:`voicelab.tts_path` にあり、ここに置くのは
OpenAI に固有の 3 つ（:class:`OpenAiTts` の送受信、料金、声の選び方）だけ。
形は :mod:`voicelab.deepgram_path` と同じにしてある。

B（ElevenLabs）との違い（これが比べたいもの）:

- TTS が ElevenLabs の ``stream-input``（WebSocket、1 本の接続に文を流し込む）から、
  OpenAI の ``POST /v1/audio/speech``（REST、**1 文 1 リクエスト**）に変わる。応答本文は
  chunked で届くのでストリーミングで読み、最初のバイトが届いた瞬間に時刻を打つ
- 出力は ``response_format=pcm``（24kHz / 16bit signed little-endian / mono、ヘッダなし）。
  B の 16kHz とはサンプルレートが違うが、**変換せずに** 24kHz の WAV として残す
  （:attr:`voicelab.tts_path.TtsSpec.sample_rate` の注記）
- モデルと声が別の指定になっている（Deepgram はモデル id が声を兼ねる）。``results/`` の
  段はモデル名（``gpt-4o-mini-tts``）で切るので、声を変えて比べるときは書き起こしの
  「声」の節で見分ける
- 費用は ElevenLabs のクレジットではなく**ドル**。``credits`` 列は 0、ドルは ``usd`` 列と
  ``note`` 列と書き起こしに書く。gpt-4o-mini-tts は**トークン課金**なので、ここで出すのは
  音の長さからの見積り（:func:`estimate_usd`）

声とモデルは 2026-09-25 に https://developers.openai.com/api/docs/guides/text-to-speech
（旧 platform.openai.com/docs/guides/text-to-speech）で確かめた。組み込みの声は 13
（alloy / ash / ballad / coral / echo / fable / nova / onyx / sage / shimmer / verse /
marin / cedar）で、「音質が一番よい」と推されているのが marin と cedar。
"""

import dataclasses
import json
import queue
import threading
import time
from typing import Callable

import httpx

from . import tts_path, tts_provider
from .config import load_env
from .custom_path import AudioSink, CustomPathError, Turn

#: OpenAI の REST の根。TTS は ``/v1/audio/speech``、接続を温めるのは ``/v1/models/<モデル>``。
API_BASE = 'https://api.openai.com'

#: CSV の ``path`` 列、CLI の ``--tts`` の値。**一度決めた綴りは変えない。**
PATH_OPENAI = 'openai'

#: ``results/openai/<モデル>_<声>/`` の会社の段。
COMPANY = 'openai'

#: TTS モデルの既定。`.env` の ``OPENAI_TTS_MODEL`` で変えられる。
#: 2026-09-25 時点で TTS は gpt-4o-mini-tts（一番新しい）/ tts-1（遅延が小さい）/ tts-1-hd の 3 つ。
DEFAULT_OPENAI_TTS_MODEL = 'gpt-4o-mini-tts'

#: 声の既定。`.env` の ``OPENAI_TTS_VOICE`` で変えられる。
#:
#: 公式のガイド（2026-09-25 確認）が「音質が一番よい」と推すのは marin と cedar の 2 つ。
#: B の声（Jessica）と Deepgram の既定（Izanami）に揃えて女性の marin にしている。
#: 組み込みの声に日本語専用のものは無い（どの声も多言語を話す）。
DEFAULT_OPENAI_TTS_VOICE = 'marin'

#: 出力の形。``pcm`` は 24kHz / 16bit signed little-endian / mono の生 PCM でヘッダなし
#: （ガイドの出力形式の節、2026-09-25 確認）。ヘッダが無いので文ごとにそのままつなげる。
RESPONSE_FORMAT = 'pcm'
SAMPLE_RATE = 24000

#: 書き起こしと dry-run に出す出力形式の名前。
OUTPUT_FORMAT = f'{RESPONSE_FORMAT}_{SAMPLE_RATE}'

#: 1 秒ぶんの PCM のバイト数（24,000 サンプル × 2 バイト × 1 チャンネル）。音の長さの換算に使う。
BYTES_PER_SECOND = SAMPLE_RATE * 2

#: gpt-4o-mini-tts の「生成した音 1 分あたり」の見積り単価（ドル）。
#:
#: 出典: https://developers.openai.com/api/docs/pricing （2026-09-25 確認）には
#: text input $0.60 / 1M tokens、audio output $12.00 / 1M tokens とトークン単価しか無い。
#: 1 分 $0.015 は 2025-03 の公開時に OpenAI が出した「推定」の値で、トークン単価は
#: 2026-09-25 の時点でも当時と同じ。openai.com/api/pricing は 2026-09-25 に 403 で読めなかった。
USD_PER_MINUTE = 0.015

#: 1 リクエストの待ち上限（秒）。接続・読み取りとも。
REQUEST_TIMEOUT_SECONDS = 30.0


class OpenAiPathError(CustomPathError):
    """1 往復を始められなかった、または続けられなかった。

    :class:`voicelab.custom_path.CustomPathError` を継いでいるのは、CLI の受け口を
    増やさずに B と同じ扱いにするため。
    """


# --------------------------------------------------------------------------- 費用


def audio_seconds(pcm: bytes) -> float:
    """受け取った PCM（24kHz / 16bit / mono）の長さ（秒）。"""
    return len(pcm) / BYTES_PER_SECOND


def estimate_usd(turn: Turn, usd_per_minute: float = USD_PER_MINUTE) -> float:
    """受け取った音の長さからドルを見積もる。

    gpt-4o-mini-tts の課金は**トークン**（送った文字のトークンと、生成した音のトークン）で、
    文字数でも秒数でもない。音のトークン数は応答に載らないので、ここでは OpenAI が
    公開時に出した「1 分あたり約 $0.015」に音の長さを掛けて近似する。短い文ほど実際の
    請求が割高になるという報告もあり、**これは見積り**。正は OpenAI の usage ダッシュボード
    （https://platform.openai.com/usage）で見る。
    """
    return audio_seconds(turn.pcm) / 60 * usd_per_minute


# --------------------------------------------------------------------------- TTS


def speech_url(*, base: str = API_BASE) -> str:
    """``POST /v1/audio/speech`` の URL。モデルと声は本文で渡す。"""
    return f'{base}/v1/audio/speech'


def warmup_url(model: str, *, base: str = API_BASE) -> str:
    """接続を温める ``GET /v1/models/<モデル>`` の URL（課金なし。鍵とモデル名の確かめも兼ねる）。"""
    return f'{base}/v1/models/{model}'


def speech_body(sentence: str, *, model: str, voice: str) -> dict:
    """``/v1/audio/speech`` に送る本文。B と違って末尾の空白は要らない（ElevenLabs 固有の決まり）。"""
    return {
        'model': model,
        'input': sentence,
        'voice': voice,
        'response_format': RESPONSE_FORMAT,
    }


def _make_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        headers={'Authorization': f'Bearer {api_key}'},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def describe_http_error(status: int, body: bytes) -> str:
    """失敗の応答を 1 行にする。``error.code``（無ければ ``error.type``）を先頭近くに出す。

    OpenAI の残高切れは ``429`` で ``code=insufficient_quota``。本文の ``message`` は長く、
    頭の 200 文字で切ると ``code`` が落ちるので、先に取り出しておく
    （:func:`is_payment_required` がこれを見る）。
    """
    text = body.decode('utf-8', errors='replace')
    code = ''
    message = text
    try:
        error = json.loads(text).get('error') or {}
        code = error.get('code') or error.get('type') or ''
        message = error.get('message') or text
    except (ValueError, AttributeError):
        pass
    head = f'HTTP {status} {code}'.rstrip()
    return f'{head}: {message[:200]}'


class OpenAiTts:
    """文が確定するたびに ``POST /v1/audio/speech`` を投げ、応答をストリーミングで受ける。

    :class:`voicelab.custom_path.TtsStream` を満たすので、B の ``run_turn`` にそのまま渡せる。
    作りは :class:`voicelab.deepgram_path.DeepgramTts` と同じで、違うのは本文の形と認証だけ。

    **1 文 1 リクエスト**にしている。REST には B の ``stream-input`` のような「1 本の接続に
    文を流し込む」口が無いため。LLM の生成を止めないよう、``send()`` は文を列に積むだけで
    すぐ戻り、送信と受信は別スレッドが**文の順に 1 本ずつ**行う（音の順番を崩さないため）。
    そのぶん 2 文目以降は前の文の音を受け終わってから投げるので、``reply_done_ms`` は
    B より伸びうる。``first_audio_ms`` には影響しない。

    ``open()`` で 1 度 ``GET /v1/models/<モデル>``（課金なし）を投げて、TCP と TLS の接続を
    ``t0`` より**前**に張っておく。B が WebSocket の接続を ``t0`` 前に済ませているのと
    揃えるため。これをしないと、最初の文の遅延に接続の時間が乗る。
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        sink: AudioSink,
        client_factory: Callable[[str], httpx.Client] = _make_client,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._api_key = api_key
        self._model = model
        self._voice = voice
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
    def voice(self) -> str:
        return self._voice

    @property
    def url(self) -> str:
        return speech_url()

    def open(self) -> 'OpenAiTts':
        """接続を温めて、送受信のスレッドを立てる。**t0 より前に呼ぶこと。**"""
        try:
            self._client = self._client_factory(self._api_key)
            response = self._client.get(warmup_url(self._model))
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OpenAiPathError(
                f'OpenAI が HTTP {exc.response.status_code} を返しました'
                f'（鍵とモデル名 {self._model} を確かめる）'
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenAiPathError(f'OpenAI に接続できませんでした: {type(exc).__name__}') from exc
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
                body = speech_body(sentence, model=self._model, voice=self._voice)
                with self._client.stream('POST', self.url, json=body) as response:
                    if response.status_code >= 400:
                        self.error = describe_http_error(response.status_code, response.read())
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

    def __enter__(self) -> 'OpenAiTts':
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


def is_payment_required(error: str | None) -> bool:
    """TTS のエラーが残高切れか。

    OpenAI は前払いのクレジットが尽きると ``429`` の ``insufficient_quota`` を返す
    （ただの速度制限も 429 なので、``code`` で見分ける）。念のため ``402`` も拾う。
    """
    if not error:
        return False
    return error.startswith('HTTP 402') or (
        error.startswith('HTTP 429') and 'insufficient_quota' in error
    )


# --------------------------------------------------------------------------- 1 往復（共通ランナーに渡すもの）


def voice_of(env: dict[str, str]) -> str:
    """``.env`` の ``OPENAI_TTS_VOICE``。無ければ :data:`DEFAULT_OPENAI_TTS_VOICE`。"""
    return env.get('OPENAI_TTS_VOICE') or DEFAULT_OPENAI_TTS_VOICE


def _make_tts(
    api_key: str, model: str, sink: AudioSink, *, voice: str = DEFAULT_OPENAI_TTS_VOICE
) -> OpenAiTts:
    # モジュールの属性を毎回引くのは、テストが ``OpenAiTts`` を差し替えられるようにするため
    return OpenAiTts(api_key=api_key, model=model, voice=voice, sink=sink)


def _pricing_line(turn: Turn) -> str:
    return (
        f'- OpenAI（見積り）: ${estimate_usd(turn):.5f}'
        f'（音 {audio_seconds(turn.pcm):.1f} 秒 × ${USD_PER_MINUTE} / 分。'
        'トークン課金なので正は usage ダッシュボード）。ElevenLabs のクレジットは 0'
    )


def _extra_transcript(tts: object) -> list[str]:
    voice = getattr(tts, 'voice', None)
    ttfb = getattr(tts, 'ttfb_ms', None) or []
    lines = ['## OpenAI の声', f'- {voice or "（不明）"}', '']
    lines += ['## OpenAI の文ごとの応答（リクエストから最初のバイトまで）']
    lines += [f'- {index}. {ms} ms' for index, ms in enumerate(ttfb, start=1)] or ['- （なし）']
    return lines


SPEC = tts_path.TtsSpec(
    path=PATH_OPENAI,
    company=COMPANY,
    name='OpenAI',
    letter='E',
    api_key_env='OPENAI_API_KEY',
    model_env='OPENAI_TTS_MODEL',
    default_model=DEFAULT_OPENAI_TTS_MODEL,
    output_format=OUTPUT_FORMAT,
    sample_rate=SAMPLE_RATE,
    make_tts=_make_tts,
    estimate_usd=estimate_usd,
    pricing_line=_pricing_line,
    is_payment_required=is_payment_required,
    payment_message=(
        '残高がありません（HTTP 429 insufficient_quota）。クレジットを足してから流し直す。'
        'results/ には何も書いていません'
    ),
    extra_transcript=_extra_transcript,
    dry_run_action='検索 → LLM をストリーミング → 文が確定するたびに /v1/audio/speech へ 1 本ずつ投げ、音を受ける',
    error_type=OpenAiPathError,
)


def spec_for(env: dict[str, str]) -> tts_path.TtsSpec:
    """``.env`` の声を織り込んだ :data:`SPEC`。

    共通ランナーの ``make_tts`` は ``(鍵, モデル, sink)`` しか受け取らないので、
    OpenAI にだけある「声」はここで閉じ込めて渡す。
    """
    voice = voice_of(env)
    return dataclasses.replace(
        SPEC,
        make_tts=lambda api_key, model, sink: _make_tts(api_key, model, sink, voice=voice),
        dry_run_action=f'{SPEC.dry_run_action}（声 {voice}）',
        # 置き場と Run.model は ``<モデル>_<声>``。OpenAI はモデルと声が別の設定なので、
        # 声を替えた計測が同じフォルダに混ざらないようにする
        folder_name=lambda _env, model: f'{model}_{voice}',
    )


def build_openai_note(scenario: dict, turn: Turn) -> str:
    """B の ``note`` にドルを足したもの。``credits`` 列が 0 なので、費用はここと ``usd`` 列で読む。"""
    return tts_path.build_usd_note(SPEC, scenario, turn)


def describe_dry_run(scenario: dict, env: dict[str, str] | None = None) -> str:
    """接続せずに、何をするつもりかと検索結果を見せる（課金なし）。"""
    env = load_env() if env is None else env
    return tts_path.describe_dry_run(spec_for(env), scenario, env)


def run_scenario(scenario: dict, *, save_audio: bool = True, env: dict[str, str] | None = None):
    """質問を 1 つ投げて 1 往復し、:class:`voicelab.metrics.Run` を返す。

    **OpenAI のクレジットと Gemini のトークンを消費する。**

    :raises OpenAiPathError: OpenAI につなげなかった、または残高切れ（429 insufficient_quota。記録しない）。
    :raises voicelab.config.ConfigError: 鍵が足りない。
    """
    env = load_env() if env is None else env
    return tts_path.run_scenario(spec_for(env), scenario, save_audio=save_audio, env=env)


def describe_balance(_env: dict[str, str]) -> str:
    """OpenAI には残高を読む公開の API が無い。そう 1 行で言う。"""
    return 'OpenAI: 残高は API から読めない（usage ダッシュボードで見る）'


PROVIDER = tts_provider.register(
    tts_provider.TtsProvider(
        key=PATH_OPENAI,
        name='OpenAI',
        label='B: 自前構成（OpenAI）',
        required_env=('OPENAI_API_KEY', 'GEMINI_API_KEY'),
        run_scenario=run_scenario,
        describe_dry_run=describe_dry_run,
        describe_balance=describe_balance,
        cost_hint='ElevenLabs クレジット 0（ドルは usd 列。音の長さからの見積り）',
        after_run_hints=(
            '声・文ごとの OpenAI の応答時間・Gemini のトークンは results/openai/<モデル>_<声>/transcripts/ にある。',
            '費用は見積り（トークン課金）。正は https://platform.openai.com/usage で見る。',
        ),
    )
)


__all__ = [
    'COMPANY',
    'DEFAULT_OPENAI_TTS_MODEL',
    'DEFAULT_OPENAI_TTS_VOICE',
    'OpenAiPathError',
    'OpenAiTts',
    'PATH_OPENAI',
    'PROVIDER',
    'SAMPLE_RATE',
    'SPEC',
    'USD_PER_MINUTE',
    'audio_seconds',
    'build_openai_note',
    'describe_balance',
    'describe_dry_run',
    'describe_http_error',
    'estimate_usd',
    'is_payment_required',
    'run_scenario',
    'spec_for',
    'speech_body',
    'speech_url',
    'voice_of',
    'warmup_url',
]


if __name__ == '__main__':
    # python -m voicelab.openai_path rollback
    # **OpenAI のクレジットと Gemini のトークンを消費する。**
    import sys

    from .config import find_scenario

    print(run_scenario(find_scenario(sys.argv[1] if len(sys.argv) > 1 else 'rollback')))
